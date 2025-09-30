import torch

X = torch.randn(5, 5)
X_T = X.T  # or X.t() or X.transpose(0, 1)

# Verify they share the same data pointer
print(X.data_ptr() == X_T.data_ptr())  # True

# They have different strides
print(f"X strides: {X.stride()}")      # (5, 1) for row-major
print(f"X_T strides: {X_T.stride()}")  # (1, 5) - columns become rows

# Modifying one affects the other since they share memory
X[0, 1] = 999
print(X_T[1, 0])  # Also 999