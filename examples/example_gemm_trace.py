#!/usr/bin/env python3
"""Trace an SM90 GEMM kernel and visualize in Perfetto.

Run with:    QUACK_TRACE=1 python examples/example_gemm_trace.py
Visualize:   Open /tmp/gemm_trace.json in https://ui.perfetto.dev
"""

import math

import torch
import cutlass
from cutlass import Float32

from quack.gemm import gemm
from quack.gemm_default_epi import GemmDefaultSm90
from quack.trace import TraceSession

M, N, K = 4096, 4096, 4096
TILE_M, TILE_N = 128, 192
CLUSTER_M, CLUSTER_N = 2, 1
OUT_PATH = "/tmp/gemm_trace.json"


def main():
    A = torch.randn(1, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(1, N, K, device="cuda", dtype=torch.float16)
    D = torch.empty(1, M, N, device="cuda", dtype=torch.float16)

    # Query the GEMM config for block size (threads_per_cta).
    g = GemmDefaultSm90(Float32, cutlass.Float16, (TILE_M, TILE_N), (CLUSTER_M, CLUSTER_N, 1))
    # grid_size = math.ceil(M / TILE_M) * math.ceil(N / TILE_N)
    grid_size = 132

    with TraceSession(OUT_PATH, grid_size=grid_size, block_size=g.threads_per_cta,
                      region_names=["tma_load", "mma", "epilogue"]) as sess:
        gemm(A, B, D, C=None, tile_count_semaphore=None,
             tile_M=TILE_M, tile_N=TILE_N,
             cluster_M=CLUSTER_M, cluster_N=CLUSTER_N,
             persistent=True, pingpong=True, trace_ptr=sess.ptr)

    # Verify correctness.
    ref = A[0] @ B[0].T
    print(f"max error: {(D[0] - ref).abs().max().item():.4f}")
    print(f"Open {OUT_PATH} in https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
