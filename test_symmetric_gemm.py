import torch
import pytest

from quack.reduction_base import torch2cute_dtype_map
from quack.symmetric_dense_gemm_sm90 import symmetric_dense_gemm

class TestSymmetricGemm:
    """Unit tests for symmetric dense GEMM wrapper."""
    
    @pytest.fixture(params=[torch.float16, torch.bfloat16, torch.float32])
    def dtype(self, request):
        """Test different data types."""
        return request.param
    
    @property
    def default_shape(self):
        """Default shape for most tests (M, K, L)."""
        return (1024, 512, 2)
    
    def torch_reference(self, a, b=None, c=None, alpha=1.0, beta=1.0):
        """Reference implementation using PyTorch operations.
        
        Args:
            a: Input tensor A of shape (M, K, L)
            b: Input tensor B of shape (M, K, L) - if None, uses A (symmetric case)
            c: Optional bias tensor C of shape (M, M, L)
            alpha: Scaling factor for A @ B^T
            beta: Scaling factor for C
            
        Returns:
            Result tensor of shape (M, M, L)
        """
        if b is None:
            b = a
            
        # Use einsum for batched matrix multiplication: A @ B^T
        # a: (M, K, L), b: (M, K, L) -> result: (M, M, L)
        result = alpha * torch.einsum('mkl,nkl->mnl', a, b)
        
        if c is not None:
            result = result + beta * c
                
        return result
    
    def create_test_tensor(self, M, K, L, dtype, device, stride_pattern="mkl", seed=None):
        """Create test tensor with specified stride pattern.
        
        Args:
            M, K, L: Tensor dimensions
            dtype: Data type
            device: Device ('cuda' or 'cpu')
            stride_pattern: How to arrange strides - 'mkl' means M has stride 1
            seed: Random seed for reproducibility
        """
        if stride_pattern == "mkl":
            # M has stride 1: (M, K, L) with strides (1, M, M*K)
            tensor = torch.empty_strided(
                (M, K, L), 
                (1, M, M*K), 
                dtype=dtype, 
                device=device
            )
        elif stride_pattern == "kml":
            # K has stride 1: (M, K, L) with strides (K, 1, M*K)
            tensor = torch.empty_strided(
                (M, K, L), 
                (K, 1, M*K), 
                dtype=dtype, 
                device=device
            )
        else:
            raise ValueError(f"Unsupported stride pattern: {stride_pattern}")
            
        # Fill with random data
        if seed is not None:
            torch.manual_seed(seed)
        tensor.uniform_(-2, 2)
        return tensor
    
    def create_symmetric_tensor(self, M, L, dtype, device, seed=None):
        """Create a symmetric tensor of shape (M, M, L)."""
        # Create with stride 1 along M dimension: (M, M, L) with strides (1, M, M*M)
        tensor = torch.empty_strided(
            (M, M, L), 
            (1, M, M*M), 
            dtype=dtype, 
            device=device
        )
        
        if seed is not None:
            torch.manual_seed(seed)
        # Fill each batch slice symmetrically
        for l in range(L):
            # Generate random upper triangular matrix
            upper = torch.triu(torch.randn(M, M, dtype=dtype, device=device))
            # Make symmetric by adding transpose
            symmetric = upper + upper.T
            tensor[:, :, l] = symmetric
            
        return tensor
    
    def test_basic_symmetric_gemm(self, dtype):
        """Test basic symmetric GEMM without bias."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        
        # Create input tensor A with stride 1 along M dimension
        a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
        
        print(f"a.shape = {a.shape}, a.stride = {a.stride()}")
        
        # Test symmetric case (B = A)
        result_quack = symmetric_dense_gemm(a, a)
        result_torch = self.torch_reference(a, a)
        
        assert result_quack.shape == result_torch.shape == (M, M, L)

        if dtype == torch.float32:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-4, rtol=1e-4)
        else:  # float16, bfloat16
            torch.testing.assert_close(result_quack, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_symmetric_gemm_with_bias(self, dtype):
        """Test symmetric GEMM with bias tensor C."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        
        # Create input tensors
        a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
        c = self.create_symmetric_tensor(M, L, dtype, device, seed=123)
        
        # Compute with our wrapper
        result_quack = symmetric_dense_gemm(a, a, c=c)
        
        # Compute reference
        result_torch = self.torch_reference(a, a, c=c)
        
        # Check shapes match
        assert result_quack.shape == result_torch.shape == (M, M, L)
        
        # Check values match
        if dtype == torch.float32:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-4, rtol=1e-4)
        else:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_alpha_beta_scaling(self, dtype):
        """Test alpha and beta scaling factors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        alpha, beta = 2.5, 0.5
        
        # Create input tensors
        a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
        c = self.create_symmetric_tensor(M, L, dtype, device, seed=123)
        
        # Compute with our wrapper
        result_quack = symmetric_dense_gemm(a, a, c=c, alpha=alpha, beta=beta)
        
        # Compute reference
        result_torch = self.torch_reference(a, a, c=c, alpha=alpha, beta=beta)
        
        # Check values match
        if dtype == torch.float32:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-4, rtol=1e-4)
        else:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_symmetry_property(self, dtype):
        """Test that output is actually symmetric (D = D^T)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        
        # Create input tensor
        a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
      
        # Compute symmetric GEMM
        result = symmetric_dense_gemm(a, a)
        
        # Check symmetry for each batch
        for l in range(L):
            matrix = result[:, :, l]
            torch.testing.assert_close(matrix, matrix.T, atol=1e-6, rtol=1e-6)
    
    def test_different_stride_patterns(self, dtype):
        """Test different tensor stride patterns."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        
        # Test both stride patterns
        for stride_pattern in ["mkl", "kml"]:
            a = self.create_test_tensor(M, K, L, dtype, device, stride_pattern, seed=42)
            
            result = symmetric_dense_gemm(a, a)
            expected = self.torch_reference(a, a)
            
            assert result.shape == (M, M, L)
            if dtype == torch.float32:
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
            else:
                torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    
    def test_different_sizes(self):
        """Test various matrix sizes to ensure robustness."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = 'cuda'
        dtype = torch.float16
        
        test_sizes = [
            (128, 128, 3),  
            (256, 256, 5),  
            (1024, 1024, 5), 
            (2048, 2048, 3),  
            (4096, 4096, 1),
        ]
        
        for M, K, L in test_sizes:
            a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
    
            result = symmetric_dense_gemm(a, a)
            expected = self.torch_reference(a, a)
            
            assert result.shape == (M, M, L)
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
            
            # Verify symmetry
            for l in range(L):
                matrix = result[:, :, l]
                torch.testing.assert_close(matrix, matrix.T, atol=1e-6, rtol=1e-6)

    def test_non_symmetric_inputs(self, dtype):
        """Test with different A and B matrices (non-symmetric case)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = self.default_shape
        device = 'cuda'
        
        # Create different A and B tensors
        a = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=42)
        b = self.create_test_tensor(M, K, L, dtype, device, "mkl", seed=123)
        
        result = symmetric_dense_gemm(a, b)
        expected = self.torch_reference(a, b)
        
        assert result.shape == (M, M, L)
        if dtype == torch.float32:
            torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        else:
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


def run_tests():
    """Run all tests manually (for debugging)."""
    test_class = TestSymmetricGemm()
    
    try:
        # Test basic functionality
        print("Testing basic symmetric GEMM...")
        test_class.test_basic_symmetric_gemm(torch.float16)
        print("✓ Basic test passed")
        
        # Test with bias
        print("Testing with bias...")
        test_class.test_symmetric_gemm_with_bias(torch.float16)
        print("✓ Bias test passed")
        
        # Test scaling
        print("Testing alpha/beta scaling...")
        test_class.test_alpha_beta_scaling(torch.float16)
        print("✓ Scaling test passed")
        
        # Test symmetry
        print("Testing symmetry property...")
        test_class.test_symmetry_property(torch.float16)
        print("✓ Symmetry test passed")

        # Test different stride patterns
        print("Testing different stride patterns...")
        test_class.test_different_stride_patterns(torch.float16)
        print("✓ Stride patterns test passed")

        # Test different sizes
        print("Testing different sizes...")
        test_class.test_different_sizes()
        print("✓ Different sizes test passed")
        
        # Test non-symmetric inputs
        print("Testing non-symmetric inputs...")
        test_class.test_non_symmetric_inputs(torch.float16)
        print("✓ Non-symmetric inputs test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests()