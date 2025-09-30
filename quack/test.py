import torch

def test_unsqueeze_preserves_data_ptr():
    """Test that conditional unsqueeze preserves data pointer."""
    
    # Test case 1: 2D tensors (will be unsqueezed)
    preact_out_2d = torch.randn(5, 5)
    postact_out_2d = torch.randn(5, 5)
    
    original_preact_ptr = preact_out_2d.data_ptr()
    original_postact_ptr = postact_out_2d.data_ptr()
    
    # Simulate the conditional logic
    if preact_out_2d.ndim == 2:
        D = preact_out_2d.unsqueeze(0)
    else:
        D = preact_out_2d
    
    if postact_out_2d.ndim == 2:
        PostAct = postact_out_2d.unsqueeze(0)
    else:
        PostAct = postact_out_2d
    
    # Verify data pointers are the same
    assert D.data_ptr() == original_preact_ptr, "D should share data with preact_out"
    assert PostAct.data_ptr() == original_postact_ptr, "PostAct should share data with postact_out"
    assert D.data_ptr() == preact_out_2d.data_ptr(), "D and preact_out_2d should share data"
    assert PostAct.data_ptr() == postact_out_2d.data_ptr(), "PostAct and postact_out_2d should share data"
    
    print("✓ 2D case: Data pointers preserved after unsqueeze")
    print(f"  preact_out shape: {preact_out_2d.shape} -> D shape: {D.shape}")
    print(f"  Data ptr: {original_preact_ptr} == {D.data_ptr()}")
    
    # Test case 2: 3D tensors (will NOT be unsqueezed)
    preact_out_3d = torch.randn(2, 5, 5)
    postact_out_3d = torch.randn(2, 5, 5)
    
    original_preact_ptr_3d = preact_out_3d.data_ptr()
    original_postact_ptr_3d = postact_out_3d.data_ptr()
    
    if preact_out_3d.ndim == 2:
        D_3d = preact_out_3d.unsqueeze(0)
    else:
        D_3d = preact_out_3d
    
    if postact_out_3d.ndim == 2:
        PostAct_3d = postact_out_3d.unsqueeze(0)
    else:
        PostAct_3d = postact_out_3d
    
    # Verify data pointers are the same (trivially, since no unsqueeze happened)
    assert D_3d.data_ptr() == original_preact_ptr_3d
    assert PostAct_3d.data_ptr() == original_postact_ptr_3d
    assert D_3d is preact_out_3d, "D_3d should be the same object as preact_out_3d"
    assert PostAct_3d is postact_out_3d, "PostAct_3d should be the same object as postact_out_3d"
    
    print("✓ 3D case: Data pointers preserved (no unsqueeze needed)")
    print(f"  preact_out shape: {preact_out_3d.shape} -> D shape: {D_3d.shape}")
    
    # Test case 3: Verify transpose case from your code
    lower_triangle = torch.randn(5, 5)
    upper_triangle = lower_triangle.transpose(-1, -2)
    
    assert lower_triangle.data_ptr() == upper_triangle.data_ptr(), \
        "Transpose should share data pointer"
    print("✓ Transpose case: Data pointers are identical")
    print(f"  lower: {lower_triangle.data_ptr()} == upper: {upper_triangle.data_ptr()}")
    
    # Test case 4: Combined unsqueeze + transpose
    lower_2d = torch.randn(5, 5)
    upper_2d = lower_2d.transpose(-1, -2)
    
    if lower_2d.ndim == 2:
        lower_3d = lower_2d.unsqueeze(0)
        upper_3d = upper_2d.unsqueeze(0)
    else:
        lower_3d = lower_2d
        upper_3d = upper_2d
    
    assert lower_3d.data_ptr() == upper_3d.data_ptr() == lower_2d.data_ptr()
    print("✓ Combined case: Unsqueeze + transpose preserves shared data pointer")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_unsqueeze_preserves_data_ptr()