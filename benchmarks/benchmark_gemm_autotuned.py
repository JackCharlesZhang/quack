#!/usr/bin/env python3
"""Demo: autotuned GEMM with parallel pre-compilation.

Shows how the autotuner pre-compiles all tile configs in parallel via
persistent subprocess workers, then benchmarks each to pick the fastest.

Usage:
    # Default 8192x8192x8192 with verbose output
    python benchmarks/benchmark_gemm_autotuned.py

    # Custom size
    python benchmarks/benchmark_gemm_autotuned.py --M 8192 --N 8192 --K 4096

    # Control worker count
    QUACK_COMPILE_WORKERS=8 python benchmarks/benchmark_gemm_autotuned.py

    # Cold start: clear .so + autotuning caches to see parallel pre-compilation
    python benchmarks/benchmark_gemm_autotuned.py --cold

Environment variables:
    QUACK_COMPILE_WORKERS   Number of parallel compile workers (default: 4)
    QUACK_PRINT_AUTOTUNING  Set to "1" for verbose autotuner output (always on in this script)
"""

import argparse
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from quack.autotuner import default_cache_dir
from quack.cache_utils import get_cache_path
from quack.gemm_interface import gemm, gemm_act


def clear_caches():
    """Clear both the .so kernel cache and the autotuning result cache."""
    # .so kernel cache (from cache_utils.get_cache_path)
    so_cache = str(get_cache_path())
    if os.path.isdir(so_cache):
        shutil.rmtree(so_cache)
        print(f"Cleared .so cache: {so_cache}")
    # Autotuning result cache (from autotuner.default_cache_dir)
    autotune_cache = str(default_cache_dir())
    if os.path.isdir(autotune_cache):
        shutil.rmtree(autotune_cache)
        print(f"Cleared autotuning cache: {autotune_cache}")


def tflops(flops, ms):
    return flops / (ms * 1e9)


def benchmark_gemm(m, n, k, dtype=torch.bfloat16, repeats=30):
    """Benchmark plain GEMM: out = A @ B"""
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    # gemm() expects B as (K, N)
    b = torch.randn(k, n, device="cuda", dtype=dtype) / math.sqrt(k)
    nflops = 2 * m * n * k
    nbytes = (a.numel() + b.numel() + m * n) * dtype.itemsize

    fn = lambda: gemm(a, b, out_dtype=dtype)
    fn()  # warmup / autotune
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)
    gbps = nbytes / (ms * 1e6)

    w = b.T.contiguous()  # (N, K) for F.linear
    time.sleep(0.5)
    ms_pt = do_bench(lambda: F.linear(a, w), warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack: {ms:.3f}ms  {tf:.1f} TFLOPS  {gbps:.0f} GB/s")
    print(f"  cuBLAS: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def _torch_gated_act(act_fn, x, w):
    """Reference: GEMM + gated activation. x @ w.T has 2*N columns, split into gate/up."""
    preact = F.linear(x, w)
    gate = preact[..., ::2]
    up = preact[..., 1::2]
    return act_fn(gate) * up


_act_fns = {
    "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}
_gated_act_fns = {
    "swiglu": F.silu,
    "reglu": F.relu,
    "geglu": lambda x: F.gelu(x, approximate="tanh"),
}
_gated_activations = set(_gated_act_fns)


def benchmark_gemm_act(m, n, k, activation="gelu_tanh_approx", dtype=torch.bfloat16, repeats=30):
    """Benchmark fused GEMM + activation (supports both regular and gated activations)."""
    is_gated = activation in _gated_activations
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    # For gated: B is (K, 2*N) with interleaved gate/up; output is (M, N)
    b_n = 2 * n if is_gated else n
    b = torch.randn(k, b_n, device="cuda", dtype=dtype) / math.sqrt(k)
    nflops = 2 * m * b_n * k

    fn = lambda: gemm_act(a, b, activation=activation, out_dtype=dtype)
    fn()  # warmup / autotune
    time.sleep(0.5)
    ms = do_bench(fn, warmup=5, rep=repeats)
    tf = tflops(nflops, ms)

    # Baseline: torch.compile(GEMM + activation)
    w = b.T.contiguous()  # (N or 2*N, K) for F.linear
    if is_gated:
        ref_fn = torch.compile(lambda: _torch_gated_act(_gated_act_fns[activation], a, w))
    else:
        act_fn = _act_fns[activation]
        ref_fn = torch.compile(lambda: act_fn(F.linear(a, w)))
    ref_fn()  # compile warmup
    ref_fn()
    time.sleep(0.5)
    ms_pt = do_bench(ref_fn, warmup=5, rep=repeats)
    tf_pt = tflops(nflops, ms_pt)

    print(f"  quack: {ms:.3f}ms  {tf:.1f} TFLOPS")
    print(f"  cuBLAS + torch.compile: {ms_pt:.3f}ms  {tf_pt:.1f} TFLOPS")
    print(f"  speedup: {ms_pt / ms:.2f}x")
    return ms, tf


def main():
    parser = argparse.ArgumentParser(description="Demo autotuned GEMM with parallel compilation")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dim", type=int, default=4096, help="Model dim for transformer shapes")
    parser.add_argument("--batch", type=int, default=8192, help="Batch size for transformer shapes")
    parser.add_argument("--cold", action="store_true", help="Clear .so and autotuning caches first")
    args = parser.parse_args()

    if args.cold:
        clear_caches()

    os.environ["QUACK_PRINT_AUTOTUNING"] = "1"
    os.environ["QUACK_FORCE_CACHE_UPDATE"] = "1"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    M, N, K = args.M, args.N, args.K

    print(f"GEMM autotuning demo  (workers={os.environ.get('QUACK_COMPILE_WORKERS', '4')})")
    print(f"  M={M}, N={N}, K={K}, dtype={args.dtype}")
    print()

    # --- 1. Plain GEMM ---
    print("=" * 60)
    print(f"GEMM: ({M}, {K}) x ({K}, {N})")
    print("=" * 60)
    benchmark_gemm(M, N, K, dtype, repeats=args.repeats)
    print()

    # --- 2. GEMM + activation ---
    print("=" * 60)
    print(f"GEMM + GeLU: ({M}, {K}) x ({K}, {N})")
    print("=" * 60)
    benchmark_gemm_act(M, N, K, "gelu_tanh_approx", dtype, repeats=args.repeats)
    print()

    # --- 3. Transformer-relevant shapes ---
    batch = args.batch
    dim = args.dim
    head_dim = 128
    n_q_heads = dim // head_dim
    n_kv_heads = n_q_heads // 4  # GQA
    qkv_dim = (n_q_heads + 2 * n_kv_heads) * head_dim
    ffn = int(dim * 3.5)  # Llama-3 ratio

    print("=" * 60)
    print(f"Transformer shapes (batch={batch}, dim={dim})")
    print("=" * 60)
    gemm_shapes = [
        ("QKV proj", batch, qkv_dim, dim),
        ("Attn out", batch, dim, dim),
        ("FFN down", batch, dim, ffn),
    ]
    for label, m, n, k in gemm_shapes:
        print(f"\n  {label}: ({m}, {k}) x ({k}, {n})")
        benchmark_gemm(m, n, k, dtype, repeats=args.repeats)

    # FFN up: fused GEMM + SwiGLU — B is (K, 2*ffn), output is (M, ffn)
    print(f"\n  FFN up + SwiGLU: ({batch}, {dim}) x ({dim}, {2 * ffn})")
    benchmark_gemm_act(batch, ffn, dim, "swiglu", dtype, repeats=args.repeats)


if __name__ == "__main__":
    main()
