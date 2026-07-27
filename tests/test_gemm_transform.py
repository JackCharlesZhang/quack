# Copyright (c) 2026, Tri Dao.
"""Gate for the RS mainloop (mma_is_rs=True): it must be bitwise-identical to
the SS mainloop — same WGMMA instruction, same k-tile order, same accumulation
order; only the A operand source differs (ldmatrix s2r load of the fragment
vs the SS descriptor read)."""

import math

import pytest
import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack

from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_default_epi import GemmDefaultSm90
from quack.tile_scheduler import TileSchedulerOptions

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity(torch.device("cuda"))[0] != 9,
    reason="RS mainloop (mma_is_rs) is SM90 only",
)

_TORCH2CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def _run_gemm(A, B, D, tile_mnk, cluster_mnk, pingpong, mma_is_rs):
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    mA = from_dlpack(A, assumed_align=16)
    mB = from_dlpack(B, assumed_align=16)
    mD = from_dlpack(D, assumed_align=16)
    epi_args = GemmDefaultSm90.EpilogueArguments()
    scheduler_args = TileSchedulerOptions(Int32(get_max_active_clusters(math.prod(cluster_mnk))))
    gemm_obj = GemmDefaultSm90(
        Float32,
        _TORCH2CUTE[A.dtype],
        tile_mnk,
        cluster_mnk,
        pingpong=pingpong,
        mma_is_rs=mma_is_rs,
    )
    compiled = cute.compile(gemm_obj, mA, mB, mD, None, epi_args, scheduler_args, None, stream)
    compiled(mA, mB, mD, None, epi_args, scheduler_args, None, stream)


def _check_rs_vs_ss(m, n, k, tile_mnk, cluster_mnk, pingpong, a_major, dtype):
    torch.manual_seed(0)
    device = "cuda"
    if a_major == "k":
        A = torch.randn(m, k, dtype=dtype, device=device) / math.sqrt(k)
    else:
        A = (torch.randn(k, m, dtype=dtype, device=device) / math.sqrt(k)).t()
    B = torch.randn(n, k, dtype=dtype, device=device) / math.sqrt(k)
    D_ss = torch.empty(m, n, dtype=dtype, device=device)
    D_rs = torch.empty(m, n, dtype=dtype, device=device)
    _run_gemm(A, B, D_ss, tile_mnk, cluster_mnk, pingpong, mma_is_rs=False)
    _run_gemm(A, B, D_rs, tile_mnk, cluster_mnk, pingpong, mma_is_rs=True)
    torch.cuda.synchronize()
    # Sanity: the RS result is a correct GEMM at all.
    ref = (A.float() @ B.float().mT).to(dtype)
    torch.testing.assert_close(D_rs, ref, atol=3e-2, rtol=1e-3)
    # The gate: identical accumulation order => bitwise-equal outputs.
    assert torch.equal(D_rs, D_ss), "RS mainloop is not bitwise-identical to SS"


@pytest.mark.parametrize("pingpong", [False, True])
@pytest.mark.parametrize("a_major", ["k", "m"])
def test_rs_identity_bitwise(a_major, pingpong):
    # k = 4.5 k-tiles exercises the TMA zero-fill K tail through the fill.
    _check_rs_vs_ss(
        m=384,
        n=256,
        k=288,
        tile_mnk=(128, 128, 64),
        cluster_mnk=(1, 1, 1),
        pingpong=pingpong,
        a_major=a_major,
        dtype=torch.bfloat16,
    )


def test_rs_identity_fp16():
    _check_rs_vs_ss(
        m=256,
        n=256,
        k=512,
        tile_mnk=(128, 128, 64),
        cluster_mnk=(1, 1, 1),
        pingpong=False,
        a_major="k",
        dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "tile_mnk",
    [
        (64, 128, 64),  # atom (1, 1): single MMA warpgroup
        (256, 128, 64),  # atom (2, 1), 128-row warpgroup extent
        (192, 256, 64),  # atom (1, 2): N-split warpgroups + N-permuted tiled_mma
    ],
)
def test_rs_identity_atom_layouts(tile_mnk):
    _check_rs_vs_ss(
        m=384,
        n=512,
        k=256,
        tile_mnk=tile_mnk,
        cluster_mnk=(1, 1, 1),
        pingpong=False,
        a_major="k",
        dtype=torch.bfloat16,
    )


def test_rs_identity_cluster():
    # A-multicast (cluster_N = 2) writes sA; the RS s2r load reads it — orthogonal.
    _check_rs_vs_ss(
        m=512,
        n=512,
        k=256,
        tile_mnk=(128, 128, 64),
        cluster_mnk=(1, 2, 1),
        pingpong=False,
        a_major="k",
        dtype=torch.bfloat16,
    )


def test_rs_identity_short_k():
    # 1- and 2-k-tile problems exercise the mainloop's prologue/tail special
    # cases (no steady iterations; preload straight into the tail).
    for k in (64, 128):
        _check_rs_vs_ss(
            m=256,
            n=256,
            k=k,
            tile_mnk=(128, 128, 64),
            cluster_mnk=(1, 1, 1),
            pingpong=False,
            a_major="k",
            dtype=torch.bfloat16,
        )
