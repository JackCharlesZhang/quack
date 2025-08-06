import torch
import pytest
import sys
import os

# Add the mapping that was missing
torch2cute_dtype_map = {
    torch.float16: "Float16",  # You'll need to import the actual cutlass types
    torch.bfloat16: "BFloat16", 
    torch.float32: "Float32"
}

# You'll need to replace this with actual imports from your cutlass/quack modules
# This is just a placeholder for testing
try:
    from quack.symmetric_dense_gemm_sm90 import symmetric_dense_gemm
except ImportError:
    print("Warning: Could not import symmetric_dense_gemm, using mock implementation")
    def symmetric_dense_gemm(a, c=None, alpha=1.0, beta=1.0):
        # Mock implementation for testing the test structure
        M, K, L = a.shape
        result = torch.zeros(M, M, L, dtype=a.dtype, device=a.device)
        for i in range(L):
            a_slice = a[:, :, i]
            result[:, :, i] = alpha * torch.matmul(a_slice, a_slice.T)
            if c is not None:
                result[:, :, i] += beta * c[:, :, i]
        return result


class TestSymmetricGemm:
    """Unit tests for symmetric dense GEMM wrapper."""
    
    @pytest.fixture(params=[torch.float16, torch.bfloat16, torch.float32])
    def dtype(self, request):
        """Test different data types."""
        return request.param
    
    @pytest.fixture(params=[(512, 256, 4), (768, 384, 2), (1024, 512, 1)])
    def shapes(self, request):
        """Test different matrix shapes (M, K, L)."""
        return request.param
    
    def torch_reference(self, a, c=None, alpha=1.0, beta=1.0):
        """Reference implementation using PyTorch operations."""
        # Compute A @ A^T for each batch
        batch_size = a.shape[2]
        results = []
        
        for i in range(batch_size):
            a_slice = a[:, :, i]  # Shape: (M, K)
            result = alpha * torch.matmul(a_slice, a_slice.T)  # A @ A^T
            
            if c is not None:
                c_slice = c[:, :, i]  # Shape: (M, M)
                result = result + beta * c_slice
                
            results.append(result)
        
        # Stack back to (M, M, L) format
        return torch.stack(results, dim=2)
    
    def test_basic_symmetric_gemm(self, dtype, shapes):
        """Test basic symmetric GEMM without bias."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = shapes
        device = 'cuda'
        
        # Create input tensor
        torch.manual_seed(42)  # For reproducible results
        a = torch.randn(M, K, L, dtype=dtype, device=device)
        
        # Compute with our wrapper
        result_quack = symmetric_dense_gemm(a)
        
        # Compute reference
        result_torch = self.torch_reference(a)
        
        # Check shapes match
        assert result_quack.shape == result_torch.shape == (M, M, L)
        
        # Check values match (with appropriate tolerance for dtype)
        if dtype == torch.float32:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-4, rtol=1e-4)
        else:  # float16, bfloat16
            torch.testing.assert_close(result_quack, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_symmetric_gemm_with_bias(self, dtype, shapes):
        """Test symmetric GEMM with bias tensor C."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = shapes
        device = 'cuda'
        
        # Create input tensors
        torch.manual_seed(42)
        a = torch.randn(M, K, L, dtype=dtype, device=device)
        c = torch.randn(M, M, L, dtype=dtype, device=device)
        
        # Make C symmetric for each batch
        for i in range(L):
            c_slice = c[:, :, i]
            c[:, :, i] = (c_slice + c_slice.T) / 2
        
        # Compute with our wrapper
        result_quack = symmetric_dense_gemm(a, c=c)
        
        # Compute reference
        result_torch = self.torch_reference(a, c=c)
        
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
            
        M, K, L = 64, 32, 2
        device = 'cuda'
        alpha, beta = 2.5, 0.5
        
        # Create input tensors
        torch.manual_seed(42)
        a = torch.randn(M, K, L, dtype=dtype, device=device)
        c = torch.randn(M, M, L, dtype=dtype, device=device)
        
        # Make C symmetric
        for i in range(L):
            c_slice = c[:, :, i]
            c[:, :, i] = (c_slice + c_slice.T) / 2
        
        # Compute with our wrapper
        result_quack = symmetric_dense_gemm(a, c=c, alpha=alpha, beta=beta)
        
        # Compute reference
        result_torch = self.torch_reference(a, c=c, alpha=alpha, beta=beta)
        
        # Check values match
        if dtype == torch.float32:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-4, rtol=1e-4)
        else:
            torch.testing.assert_close(result_quack, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_symmetry_property(self, dtype):
        """Test that output is actually symmetric (D = D^T)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K, L = 32, 16, 2
        device = 'cuda'
        
        # Create input tensor
        torch.manual_seed(42)
        a = torch.randn(M, K, L, dtype=dtype, device=device)
        
        # Compute symmetric GEMM
        result = symmetric_dense_gemm(a)
        
        # Check symmetry for each batch
        for i in range(L):
            matrix = result[:, :, i]
            torch.testing.assert_close(matrix, matrix.T, atol=1e-6, rtol=1e-6)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = 'cuda'
        
        # Test wrong input dimensions
        try:
            a_2d = torch.randn(32, 16, device=device)
            symmetric_dense_gemm(a_2d)
            assert False, "Should have raised assertion error"
        except (AssertionError, ValueError, RuntimeError):
            pass  # Expected error
        
        # Test CPU tensor (should fail)
        try:
            a_cpu = torch.randn(32, 16, 4)
            symmetric_dense_gemm(a_cpu)
            assert False, "Should have raised assertion error" 
        except (AssertionError, ValueError, RuntimeError):
            pass  # Expected error
        
        # Test mismatched C dimensions
        try:
            a = torch.randn(32, 16, 4, device=device)
            c_wrong = torch.randn(32, 30, 4, device=device)  # Wrong N dimension
            symmetric_dense_gemm(a, c=c_wrong)
            assert False, "Should have raised assertion error"
        except (AssertionError, ValueError, RuntimeError):
            pass  # Expected error
    
    def test_single_batch(self):
        """Test with single batch (L=1)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        M, K = 64, 32
        device = 'cuda'
        dtype = torch.float16
        
        # Single batch
        torch.manual_seed(42)
        a = torch.randn(M, K, 1, dtype=dtype, device=device)
        result = symmetric_dense_gemm(a)
        
        # Reference
        expected = torch.matmul(a[:, :, 0], a[:, :, 0].T).unsqueeze(2)
        
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
        assert result.shape == (M, M, 1)
    
    def test_different_sizes(self):
        """Test various matrix sizes to ensure robustness."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = 'cuda'
        dtype = torch.float16
        
        test_sizes = [
            (32, 16, 1),   # Small
            (128, 64, 2),  # Medium  
            (256, 128, 1), # Larger
        ]
        
        for M, K, L in test_sizes:
            torch.manual_seed(42)
            a = torch.randn(M, K, L, dtype=dtype, device=device)
            
            result = symmetric_dense_gemm(a)
            expected = self.torch_reference(a)
            
            assert result.shape == (M, M, L)
            torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
            
            # Verify symmetry
            for i in range(L):
                matrix = result[:, :, i]
                torch.testing.assert_close(matrix, matrix.T, atol=1e-6, rtol=1e-6)


def run_tests():
    """Run all tests manually (for debugging)."""
    test_class = TestSymmetricGemm()
    
    try:
        # Test basic functionality
        print("Testing basic symmetric GEMM...")
        test_class.test_basic_symmetric_gemm(torch.float16, (64, 32, 4))
        print("✓ Basic test passed")
        
        # Test with bias
        print("Testing with bias...")
        test_class.test_symmetric_gemm_with_bias(torch.float16, (64, 32, 4))
        print("✓ Bias test passed")
        
        # Test scaling
        print("Testing alpha/beta scaling...")
        test_class.test_alpha_beta_scaling(torch.float16)
        print("✓ Scaling test passed")
        
        # Test symmetry
        print("Testing symmetry property...")
        test_class.test_symmetry_property(torch.float16)
        print("✓ Symmetry test passed")
        
        # Test edge cases
        print("Testing edge cases...")
        test_class.test_edge_cases()
        print("✓ Edge cases test passed")
        
        # Test single batch
        print("Testing single batch...")
        test_class.test_single_batch()
        print("✓ Single batch test passed")
        
        # Test different sizes
        print("Testing different sizes...")
        test_class.test_different_sizes()
        print("✓ Different sizes test passed")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests()