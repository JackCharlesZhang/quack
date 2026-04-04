import argparse
import time
from typing import Tuple, Type, Optional

import torch
from triton.testing import do_bench

from quack.gemm import gemm as quack_gemm

"""
Simplified GEMM benchmark using quack.gemm.gemm() directly.

Usage:
    python benchmarks/benchmark_gemm.py --mnkl 512,7168,2048,256 \
        --tile_shape_mn 256,256 --cluster_shape_mn 2,1 --persistent \
        --varlen_m --gather_A --skip_ref_check
"""


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEMM benchmark using quack.gemm.gemm()")

    parser.add_argument(
        "--mnkl", type=parse_comma_separated_ints, default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mn", type=parse_comma_separated_ints, default=(128, 256),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn", type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)], default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument("--tolerance", type=float, default=3e-02, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    parser.add_argument("--persistent", action="store_true", help="Persistent kernel")
    parser.add_argument("--dynamic_persistent", action="store_true", help="Dynamic persistent")
    parser.add_argument("--pingpong", action="store_true", help="Pingpong kernel")
    parser.add_argument("--varlen_m", action="store_true", help="Variable length M dimension")
    parser.add_argument("--varlen_k", action="store_true", help="Variable length K dimension")
    parser.add_argument("--gather_A", action="store_true", help="Gather A")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")

    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mn) != 2:
        parser.error("--tile_shape_mn must contain exactly 2 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")
    return args


def run(args):
    m, n, k, l = args.mnkl
    tile_M, tile_N = args.tile_shape_mn
    cluster_M, cluster_N = args.cluster_shape_mn
    persistent = args.persistent or args.dynamic_persistent
    varlen_m, varlen_k, gather_A = args.varlen_m, args.varlen_k, args.gather_A
    warmup, repeats = args.warmup_iterations, args.iterations
    tolerance = args.tolerance

    from quack.cute_dsl_utils import get_device_capacity
    device_capacity = get_device_capacity(torch.device("cuda"))
    if device_capacity[0] in [10, 11]:
        persistent = True

    print("Running Dense GEMM with:")
    print(f"mnkl: {args.mnkl}")
    print(f"Tile Shape: {args.tile_shape_mn}, Cluster Shape: {args.cluster_shape_mn}")
    print(f"Warmup iterations: {warmup}")
    print(f"Iterations: {repeats}")
    print(f"Skip reference checking: {args.skip_ref_check}")

    torch.manual_seed(1111)
    device = "cuda"

    # ── Tensor creation ───────────────────────────────────────────────────────
    # quack.gemm.gemm() conventions:
    #   A: (l, m, k) or (total_m, k) if varlen_m — k-major
    #   B: (l, n, k) — k-major
    #   D: (l, m, n) or (total_m, n) if varlen_m — n-major
    cu_seqlens_m, cu_seqlens_k, A_idx = None, None, None
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=device) if args.dynamic_persistent else None
    )

    if varlen_m:
        total_m = m * l
        cu_seqlens_m = torch.arange(0, l + 1, dtype=torch.int32, device=device) * m
        A = torch.randn(total_m, k, dtype=torch.bfloat16, device=device) / (k**0.5)
        if gather_A:
            A_idx = torch.randperm(total_m, dtype=torch.int32, device=device)
        B = torch.randn(l, n, k, dtype=torch.bfloat16, device=device) / (k**0.5)
        D = torch.empty(total_m, n, dtype=torch.bfloat16, device=device)
    elif varlen_k:
        total_k = k * l
        cu_seqlens_k = torch.arange(0, l + 1, dtype=torch.int32, device=device) * k
        # m-major A, n-major B for varlen_k
        A = torch.randn(total_k, m, dtype=torch.bfloat16, device=device).T
        B = torch.randn(total_k, n, dtype=torch.bfloat16, device=device).T
        D = torch.empty(l, m, n, dtype=torch.bfloat16, device=device)
    else:
        A = torch.randn(l, m, k, dtype=torch.bfloat16, device=device) / (k**0.5)
        B = torch.randn(l, n, k, dtype=torch.bfloat16, device=device) / (k**0.5)
        D = torch.empty(l, m, n, dtype=torch.bfloat16, device=device)

    # ── Run / ref check ───────────────────────────────────────────────────────
    def fn():
        quack_gemm(
            A, B, D, C=None,
            tile_count_semaphore=tile_count_semaphore,
            tile_M=tile_M, tile_N=tile_N,
            cluster_M=cluster_M, cluster_N=cluster_N,
            pingpong=args.pingpong,
            persistent=persistent,
            is_dynamic_persistent=args.dynamic_persistent,
            cu_seqlens_m=cu_seqlens_m,
            cu_seqlens_k=cu_seqlens_k,
            A_idx=A_idx,
        )
        if tile_count_semaphore is not None and varlen_m:
            tile_count_semaphore.zero_()

    if not args.skip_ref_check:
        fn()
        torch.cuda.synchronize()
        if varlen_m:
            ref = torch.cat([
                (A[A_idx[cu_seqlens_m[i]:cu_seqlens_m[i+1]]] if gather_A
                 else A[cu_seqlens_m[i]:cu_seqlens_m[i+1]]) @ B[i].T
                for i in range(l)
            ])
        elif varlen_k:
            ref = torch.stack([
                A[:, cu_seqlens_k[i]:cu_seqlens_k[i+1]] @
                B[:, cu_seqlens_k[i]:cu_seqlens_k[i+1]].T
                for i in range(l)
            ])
        else:
            ref = torch.bmm(A, B.mT)
        torch.testing.assert_close(D, ref.to(torch.bfloat16), atol=tolerance, rtol=1e-3)
        print("Ref check PASSED")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    flops = 2 * m * n * k * l
    bytes_A = m * k * l * 2
    bytes_B = n * k * l * 2
    bytes_D = m * n * l * 2
    total_bytes = bytes_A + bytes_B + bytes_D

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        fn_cublas = lambda: torch.bmm(A, B.mT)
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    time.sleep(0.5)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)
    gbps = total_bytes / (timing * 1e6)
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}, GB/s: {gbps:.0f}")
    fn()

    if not (varlen_m or varlen_k) and not gather_A:
        time.sleep(0.5)
        timing_cublas = do_bench(fn_cublas, warmup=warmup, rep=repeats)
        tflops_cublas = flops / (timing_cublas * 1e9)
        print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
    print("PASS")
