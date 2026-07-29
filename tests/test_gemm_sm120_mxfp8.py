# Copyright (c) 2026, Tri Dao.
"""SM120 MXFP8 block-scaled GEMM (warp-level MmaMXF8Op with REAL e8m0 scale
factors): numerics vs the dequantized reference, and bit-exact vs cuBLAS.

Scope (first cut): K-major A and B, fp8 e4m3/e5m2 with e8m0 scales
(sf_vec_size 32). fp4/fp6 formats and varlen_k (m-major A) are rejected at
validation. Unlike the plain-fp8 unit-scale fast path (which falls back to
MmaFP8Op on the H100 CI proxy), real scale factors REQUIRE the sm_120a
kind::mxf8f6f4 block_scale instruction, so these tests run on SM120 only.
"""

import pytest
import torch
import torch.nn.functional as F

from quack.blockscaled.operand import BlockScaledFormat, BlockScaledOperand
from quack.blockscaled.utils import blockscaled_quantize, scale_blocked_for_cublas
from quack.cute_dsl_utils import get_device_capacity
from quack.gemm_config import GemmConfig
from quack.gemm_interface import (
    _prep_blockscaled,
    _sf_batch_canonicalize,
    _unpack_operand,
    gemm,
    gemm_act,
    gemm_add,
    gemm_add_inplace,
    gemm_blockscaled_ref,
    gemm_tuned,
)

_ARCH = get_device_capacity(torch.device("cuda"))[0] if torch.cuda.is_available() else 0
requires_sm120 = pytest.mark.skipif(_ARCH != 12, reason="SM120 blockscaled warp-MMA path")


def _quantized_operands(fmt, m, n, k, batched=False, seed=0):
    torch.manual_seed(seed)
    L = 2 if batched else 1
    shape_a = (L, m, k) if batched else (m, k)
    shape_w = (L, n, k) if batched else (n, k)
    a_hp = torch.randn(*shape_a, device="cuda", dtype=torch.bfloat16) * k**-0.5
    w_hp = torch.randn(*shape_w, device="cuda", dtype=torch.bfloat16) * k**-0.5
    qa, sfa = blockscaled_quantize(a_hp, fmt)
    qw, sfw = blockscaled_quantize(w_hp, fmt)
    fmt_obj = BlockScaledFormat.from_name(fmt)
    A = BlockScaledOperand.from_parts(qa, sfa, fmt_obj)
    W = BlockScaledOperand.from_parts(qw, sfw, fmt_obj)
    return A, W.mT  # B = (K, N) logical view; qdata stride-swap, scale unchanged


def _gemm_with_config(A, B, config=None, split_k=1):
    """gemm(A, B, tuned=False) with a forced GemmConfig (the public wrapper
    exposes no config knob)."""
    opA, opB = _unpack_operand(A), _unpack_operand(B)
    Ad, Bd = opA.data, opB.data
    SFA, SFB, bs_format_a, bs_format_b = _prep_blockscaled(opA, opB)
    SFA, SFB = _sf_batch_canonicalize(SFA, SFB, Ad.ndim == 3)
    out_shape = (
        (Ad.shape[0], Bd.shape[-1]) if Ad.ndim == 2 else (Ad.shape[0], Ad.shape[-2], Bd.shape[-1])
    )
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=Ad.device)
    gemm_tuned.fn(
        Ad,
        Bd,
        out,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        config=config,
        split_k=split_k,
    )
    return out


def _sm120_config(tile_m, tile_n, pingpong=False):
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=1,
        cluster_n=1,
        pingpong=pingpong,
        is_dynamic_persistent=True,
        device_capacity=12,
    )


def _rel_err(out, ref):
    return (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()


@requires_sm120
@pytest.mark.parametrize("fmt", ["mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "shape_mnk",
    [
        (256, 256, 256),
        (128, 128, 256),
        (448, 320, 512),  # M, N not multiples of 128 (padded SF rows)
        (1024, 256, 8192),  # long K: SF pipeline over many k-tiles
    ],
)
def test_sm120_mxfp8_gemm(fmt, batched, shape_mnk):
    m, n, k = shape_mnk
    A, B = _quantized_operands(fmt, m, n, k, batched)
    out = gemm(A, B, tuned=False)
    ref = gemm_blockscaled_ref(A, B)
    expected_shape = (2, m, n) if batched else (m, n)
    assert out.shape == expected_shape and out.dtype == torch.bfloat16
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"{fmt} {shape_mnk} batched={batched}: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize(
    "tile_mn,pingpong",
    [
        ((128, 128), True),
        ((256, 128), False),
        ((128, 256), False),
        ((256, 256), False),
    ],
)
def test_sm120_mxfp8_tiles(tile_mn, pingpong):
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = _gemm_with_config(A, B, config=_sm120_config(*tile_mn, pingpong=pingpong))
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"tile={tile_mn} pingpong={pingpong}: rel_err={rel}"


@requires_sm120
def test_sm120_mxfp8_split_k():
    m, n, k = 128, 128, 4096
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = _gemm_with_config(A, B, split_k=2)
    ref = gemm_blockscaled_ref(A, B)
    rel = _rel_err(out, ref)
    assert rel < 5e-3, f"split_k=2: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("seqlens_m", [[128, 128, 128], [100, 200, 150], [1, 128, 127, 129]])
def test_sm120_mxfp8_varlen_m(seqlens_m):
    """Grouped (varlen_m) MXFP8 GEMM: SFA is a single M-padded buffer
    (tile-aligned per-batch padding, batch dim 1); SFB stays per-expert."""
    import cutlass

    from quack.blockscaled.utils import create_blockscaled_varlen_m_operands

    num_experts = len(seqlens_m)
    n, k = 256, 256
    torch.manual_seed(0)
    a_ref_dq, b_ref_dq, qa, qb, a_sc, b_sc, cu_seqlens_m = create_blockscaled_varlen_m_operands(
        num_experts, 0, n, k, 32, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, seqlens_m=seqlens_m
    )
    B = qb.permute(2, 1, 0)  # (n, k, L) -> (L, K, N) with K contiguous
    A_op = BlockScaledOperand.from_parts(qa, a_sc, "mxfp8")
    B_op = BlockScaledOperand.from_parts(B, b_sc, "mxfp8", quant_dim=-2)
    out = gemm(A_op, B_op, cu_seqlens_m=cu_seqlens_m, tuned=False)

    cu = cu_seqlens_m.tolist()
    ref = torch.cat([a_ref_dq[cu[i] : cu[i + 1]] @ b_ref_dq[i].T for i in range(num_experts)])
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"varlen_m seqlens_m={seqlens_m} max_err={err}"


@requires_sm120
def test_sm120_mxfp8_vs_cublas():
    """Bit-exact comparison against torch._scaled_mm (cuBLAS MXFP8 path).
    Both consume the same fp8 values and e8m0 scales with f32 accumulation, so
    any scale mis-application (wrong k-block, wrong row, stale smem stage)
    shows up as a hard mismatch."""
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    out = gemm(A, B, tuned=False)
    sfa_flat = scale_blocked_for_cublas(A.scale.unsqueeze(0), m, k // 32)
    sfw_flat = scale_blocked_for_cublas(B.scale.unsqueeze(0), n, k // 32)
    out_cublas = torch._scaled_mm(
        A.qdata, B.qdata, scale_a=sfa_flat, scale_b=sfw_flat, out_dtype=torch.bfloat16
    )
    assert torch.equal(out, out_cublas), (
        f"quack != cuBLAS: max_err={(out.float() - out_cublas.float()).abs().max().item()}"
    )


@requires_sm120
def test_sm120_mxfp8_gemm_add():
    """Epilogue frontend (alpha/beta + C) with blockscaled operands."""
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    C = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    alpha, beta = 0.5, 2.0
    out = gemm_add(A, B, C, alpha=alpha, beta=beta, tuned=False)
    ref = alpha * gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + beta * C.float()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"gemm_add: rel_err={rel}"

    acc = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    ref2 = gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + acc.float()
    gemm_add_inplace(A, B, acc, tuned=False)
    rel = (acc.float() - ref2).abs().max().item() / ref2.abs().max().item()
    assert rel < 5e-3, f"gemm_add_inplace: rel_err={rel}"


@requires_sm120
def test_sm120_mxfp8_gemm_add_tuned():
    """Autotuned path: sweeps the blockscaled_config_ok-pruned SM120 space."""
    m, n, k = 512, 512, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    C = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out = gemm_add(A, B, C, alpha=0.5, beta=2.0)
    ref = 0.5 * gemm_blockscaled_ref(A, B, out_dtype=torch.float32) + 2.0 * C.float()
    rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 5e-3, f"tuned gemm_add: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("activation", ["relu", "gelu_tanh_approx"])
def test_sm120_mxfp8_gemm_act(activation):
    """gemm_act with blockscaled A/B, checked against the dequant reference."""
    m, n, k = 256, 256, 512
    A, B = _quantized_operands("mxfp8_e4m3", m, n, k)
    act_fn = {
        "relu": F.relu,
        "gelu_tanh_approx": lambda x: F.gelu(x, approximate="tanh"),
    }[activation]
    preact, postact = gemm_act(A, B, activation=activation, tuned=False)
    ref_post = act_fn(gemm_blockscaled_ref(A, B, out_dtype=torch.float32))
    rel = (postact.float() - ref_post).abs().max().item() / max(ref_post.abs().max().item(), 1e-6)
    assert rel < 5e-3, f"gemm_act {activation}: rel_err={rel}"


@requires_sm120
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_sm120_fp4_rejected(fmt):
    """SM120 blockscaled is MXFP8-only for now; fp4 formats must be rejected
    at validation with a legible error, not at kernel compile."""
    m, n, k = 256, 256, 256
    A, B = _quantized_operands(fmt, m, n, k)
    with pytest.raises(AssertionError, match="8-bit"):
        gemm(A, B, tuned=False)
