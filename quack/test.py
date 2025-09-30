import torch
from quack.gemm_interface import gemm_symmetric

# Create input tensors
A = torch.randn(size=(2, 8192, 4096)).to(torch.bfloat16)
B = torch.randn(size=(2, 4096, 8192)).to(torch.bfloat16)
C = torch.ones(size=(2, 8192, 8192)).to(torch.bfloat16)

# Move tensors to GPU for meaningful profiling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
A = A.to(device)
B = B.to(device)
C = C.to(device)

# Reduce profiling overhead - shorter trace
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CUDA,  # Only CUDA, not CPU
    ],
    record_shapes=False,  # Disable shape recording
    profile_memory=False,  # Disable memory profiling
    with_stack=False      # Disable stack traces
) as prof:
    
    D = gemm_symmetric(A, B, C=C.clone(), alpha=2.0, beta=3.0)
    D_ref = torch.baddbmm(C.clone(), A, B, alpha=2.0, beta=3.0)

# Export Chrome trace
prof.export_chrome_trace("gemm_profile.json")

print("Profile saved to gemm_profile.json")
print("Open chrome://tracing in Chrome and load the file to visualize")

