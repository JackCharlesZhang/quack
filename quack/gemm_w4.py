# Copyright (c) 2026, Tri Dao.
"""Weight-only-quantized GEMM for SM90: out[M, N] = act[M, K] @ dequant(W)[N, K]^T.

The packed weights are the WGMMA A operand (RS, decoded to bf16 in registers
by a :class:`~quack.operand_transform.TransformAW4`), the bf16 activations
are B, and the output is written transposed (D is (N, M) m-major = out
row-major). See quack/blockscaled/decode_formats.py for the formats and
quack/blockscaled/nvfp4_utils.py for the repack layout.

Scale-factor-strip formats (nvfp4, int4*, int4awq, mxfp4, mxfp8) pass their
repacked SF blob as ``sf``; it rides the aux A-side operand. Strip-free
formats (qtip*, int8/fp8 with the per-channel scale left to the caller,
fn-authored formats) take ``sf=None``. Per-tensor weight scales ride the
epilogue alpha.

This wrapper is thin sugar over the generic host layer (quack.gemm_host):
transforms are a first-class axis there — ``build_gemm_epi_plan(...,
transform_a=...)`` — so W4 kernels share the jit/disk cache, async compile,
and the EpiOp argument machinery with every epilogue variant. What remains
here is W4's own host surface: the offline ``prepare`` step, validation, the
measured config rule (:func:`_pick_w4_cfg`), and the split-k buffer reuse.
"""

from typing import Optional

import torch
from torch import Tensor

from quack.blockscaled.decode_formats import decode_format
from quack.cute_dsl_utils import get_device_capacity
from quack.epi_ops import ColVecLoad, RowVecLoad
from quack.gemm import _split_k_buffers
from quack.gemm_config import SplitKMode
from quack.gemm_default_epi import GemmDefaultSm90
from quack.gemm_epilogue import gemm_epilogue
from quack.gemm_host import build_gemm_epi_plan, run_gemm_epi_plan
from quack.gemm_tvm_ffi_utils import tensor_key
from quack.operand_transform.host import w4_operand_views

__all__ = ["gemm_w4a16", "gemm_w4a8", "prepare_w4_weight", "quantize_act_per_token_fp8"]

_plan_cache = {}
_splitk_buf_cache = {}


def prepare_w4_weight(q, sf=None, wformat="qtip2s"):
    """One-time weight prep: quantized weights (+ scales, format-dependent)
    -> repacked blob pair. N is padded to a multiple of 128 (tile
    granularity); bytes are shuffled into WGMMA A-fragment order for the
    in-register decode."""
    return decode_format(wformat).prepare(q, sf)


def _pick_tile_n(m_act: int) -> int:
    for cand in (16, 32, 64, 128):
        if m_act <= cand:
            return cand
    return 192


def _pick_w4_cfg(m_act: int, n_full: int, k_tiles: int) -> tuple:
    """(tile_m, tile_n, split_k). Measured invariant (H100, int4/qtip, incl.
    the machete faceoff): every winning config puts the grid at ~112-128 CTAs
    with the LARGEST tile that gets there — tile_m=128 beats 64 by 10-25% at
    equal CTA counts (2x TMA boxes, half the per-k-tile pipeline overhead per
    byte), tile_n is the largest with under half a tile of padding on m, and
    serial split-k makes up remaining grid coverage when each split keeps
    >= ~24 k-tiles (and tile_n <= 128: the f32 finalize round-trip scales
    with tile area). Prefill (m > 256): (128, 256, 1)."""
    if n_full % 128 != 0:
        tn = _pick_tile_n(m_act) if m_act <= 128 else 192
        mt = -(-m_act // tn)
        sk = 2 if (m_act <= 32 and (n_full // 64) * mt < 128 and k_tiles >= 32) else 1
        return 64, tn, sk
    if m_act > 256:
        return 128, 256, 1
    n128 = n_full // 128
    for tn in (256, 128, 64, 32, 16):
        if tn >= 2 * m_act and tn > 16:
            continue  # half the tile or more would be padding
        mt = -(-m_act // tn)
        for sk in (1, 2, 4):
            if sk > 1 and (tn > 128 or k_tiles // sk < 24):
                break
            if n128 * mt * sk >= 112:
                return 128, tn, sk
    # coverage unreachable under the tile_m=128 constraints (small N, short K):
    # fall back to 64-row tiles with the plain starvation rule
    tn = _pick_tile_n(m_act)
    mt = -(-m_act // tn)
    sk = 2 if ((n_full // 64) * mt < 128 and k_tiles >= 32) else 1
    return 64, tn, sk


def gemm_w4a16(
    act: Tensor,  # (M, K) bf16, K-major
    blob: Tensor,  # (N/64, K/tile_k, 128, 4|8 B * tile_k/64) from fmt.prepare
    sf: Optional[Tensor] = None,  # repacked SF blob from fmt.prepare (strip formats)
    tensor_scale: float = 1.0,  # per-tensor weight scale, applied as epilogue alpha
    out: Optional[Tensor] = None,  # (M, N_out) bf16
    n_out: Optional[int] = None,  # unpadded N (defaults to blob's padded N)
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cluster_n: int = 1,
    max_swizzle_size: int = 8,
    use_pdl: bool = True,
    wformat="qtip2s",  # W4_FORMATS name or DecodeFormat instance
    split_k: Optional[int] = None,  # None = auto: 2 when the grid starves the machine
) -> Tensor:
    fmt = decode_format(wformat)
    tk = fmt.tile_k
    assert act.dtype == torch.bfloat16 and act.is_contiguous()
    m_act, k = act.shape
    g, kt = blob.shape[:2]
    n_full = g * 64
    assert kt * tk == k, f"K mismatch: act K={k}, blob K={kt * tk}"
    if n_out is None:
        n_out = n_full
    auto_tm, auto_tn, auto_sk = _pick_w4_cfg(m_act, n_full, k // tk)
    if tile_m is None and tile_n is None and split_k is None:
        tile_m, tile_n, split_k = auto_tm, auto_tn, auto_sk
    if tile_m is None:
        tile_m = auto_tm
    if tile_n is None:
        tile_n = auto_tn
    assert n_full % tile_m == 0, f"padded N ({n_full}) must be divisible by tile_m ({tile_m})"
    if out is None:
        out = torch.empty(m_act, n_full, dtype=torch.bfloat16, device=act.device)
    else:
        assert out.shape == (m_act, n_out) and out.dtype == torch.bfloat16
        assert n_out == n_full, "padded N requires an internally allocated out"

    if split_k is None:
        # explicitly-tiled callers get the plain grid-starvation rule
        n_ctas = (n_full // tile_m) * ((m_act + tile_n - 1) // tile_n)
        split_k = 2 if (n_ctas < 128 and k // tk >= 32) else 1

    mA = w4_operand_views(fmt, blob, sf, tile_m)  # (blob, strip) bundle
    # out crosses caller-oriented (M_act, N_full) row-major; the trace
    # relabels it to the kernel's (N_full, M_act) m-major D (cd_transposed)
    epi_values = {"alpha": tensor_scale}

    key = (
        tensor_key(act),
        tensor_key(blob),
        tensor_key(sf),
        fmt.name if isinstance(wformat, str) else id(fmt),
        tile_m,
        tile_n,
        cluster_n,
        max_swizzle_size,
        use_pdl,
        split_k,
        tensor_scale != 1.0,
        act.device.index,
    )
    plan = _plan_cache.get(key)
    if plan is None:
        plan = build_gemm_epi_plan(
            GemmDefaultSm90,
            get_device_capacity(act.device),
            mA,
            act,
            out,
            None,
            epi_values=epi_values,
            tile_M=tile_m,
            tile_N=tile_n,
            tile_K=tk,
            cluster_M=1,
            cluster_N=cluster_n,
            max_swizzle_size=max_swizzle_size,
            transform_a=wformat if isinstance(wformat, str) else fmt,
            post_init_attrs=(() if use_pdl else (("use_pdl", False),)),
            split_k=split_k,
        )
        _plan_cache[key] = plan
    bufs = None
    if split_k > 1:
        # serial split-k turnstile + partials workspace; the kernel leaves the
        # semaphore reset, so the buffers are cached and reused across calls
        buf_key = (n_full, m_act, tile_m, tile_n, cluster_n, act.device.index)
        bufs = _splitk_buf_cache.get(buf_key)
        if bufs is None:
            bufs = _split_k_buffers(
                out.t()[None], SplitKMode.SERIAL, tile_m, tile_n, 1, cluster_n, False
            )
            _splitk_buf_cache[buf_key] = bufs
    run_gemm_epi_plan(plan, mA, act, out, None, epi_values, split_k_buffers=bufs)
    if n_out != n_full:
        out = out[:, :n_out]
    return out


# The per-token activation scale is a k-invariant per-output-row factor: it
# commutes out of the GEMM sum and applies as an exact fp32 colvec multiply in
# the epilogue (the pin flips to the kernel's rowvec — D is transposed).
@gemm_epilogue(ops={"v": ColVecLoad("v")})
def _w4a8_token_scale(acc, v):
    return {"D": acc * v}


# Folded (no-drain) W4A8: the group scale rode the decode LUT; what remains is
# the per-token scale and the fold's per-weight-channel normalizer, both fp32.
@gemm_epilogue(ops={"v": ColVecLoad("v"), "cs": RowVecLoad("cs")})
def _w4a8_folded_scale(acc, v, cs):
    return {"D": acc * v * cs}


def _pick_w4a8_cfg(m_act: int, n_full: int) -> tuple:
    """(tile_m, tile_n, split_k). Measured (H100, N=K=8192, AI/bench_w4a8.py):
    slow accum flips the W4A16 preferences. 64-row tiles win every decode
    shape through m=128 (the doubled accumulator makes 128-row consumers
    register-heavy; occupancy-2 64-row tiles hide the decode+promote latency
    better: m=1 21.0 vs 23.6us, m=64 22.1 vs 29.3), with tile_n covering m in
    one tile column. 128-row wins from m=256 (128x128 37.8 vs 64x128 44.5);
    prefill caps tile_n at 192 (2*96+16 regs — 256-wide tiles spill).
    split-k always loses here (the fp32 finalize round-trip stacks on the
    promote: m=1 sk2 24.9 vs sk1 21.0)."""
    if m_act <= 128:
        tm, tn = 64, _pick_tile_n(m_act)
    elif m_act <= 256:
        tm, tn = 128, 128
    else:
        tm, tn = 128, 192
    if n_full % tm != 0:
        tm = 64
    return tm, tn, 1


def quantize_act_per_token_fp8(act: Tensor):
    """(M, K) float -> e4m3 (M, K) + fp32 per-token scales (M,): amax/448."""
    a = act.float()
    scale = (a.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (a / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def gemm_w4a8(
    act: Tensor,  # (M, K): e4m3 K-major (pass act_scale), or float (quantized here)
    blob: Tensor,  # from decode_format(wformat).prepare / repack_w4a8_weight
    sf: Tensor,  # repacked SF strip from prepare / repack_w4a8_sf
    act_scale: Optional[Tensor] = None,  # (M,) fp32 per-token scales
    chan_scale: Optional[Tensor] = None,  # (N,) fp32 (int4smf: from fold_int4sm_scales)
    out: Optional[Tensor] = None,  # (M, N_out) bf16
    n_out: Optional[int] = None,
    tile_m: Optional[int] = None,
    tile_n: Optional[int] = None,
    cluster_n: int = 1,
    max_swizzle_size: int = 8,
    split_k: Optional[int] = None,
    wformat: str = "int4sm",
) -> Tensor:
    """W4A8: sign-magnitude int4-g128 weights x e4m3 per-token activations.

    wformat "int4sm" (exact): fp32 promotion per k-tile — no weight
    requantization; the only losses are the activation e4m3 cast and the fp8
    WGMMA accumulator. wformat "int4smf" (folded, no-drain): the group scale
    folds into the decode's e4m3 LUT (one e4m3 rounding per weight, ~2^-4
    rel worst case) and the mainloop runs fully pipelined fast-accum fp8 —
    faster at prefill; pass the fold's ``chan_scale``."""
    fmt = decode_format(wformat)
    assert fmt.mma_dtype.width == 8, f"{wformat!r} is not a W4A8 format"
    folded = not fmt.promote
    assert (chan_scale is not None) == folded, "chan_scale iff the folded format"
    if act.dtype != torch.float8_e4m3fn:
        assert act_scale is None, "act_scale is derived when act is not already e4m3"
        act, act_scale = quantize_act_per_token_fp8(act)
    assert act.is_contiguous()
    assert act_scale is not None and act_scale.dtype == torch.float32
    m_act, k = act.shape
    g, kt = blob.shape[:2]
    n_full = g * 64
    assert kt * 128 == k, f"K mismatch: act K={k}, blob K={kt * 128}"
    if n_out is None:
        n_out = n_full
    if folded and m_act > 128:
        # no promote -> no doubled accumulator: the W4A16 coverage rule holds
        # (tile_n 256 prefill); decode shapes stay on the measured 64-row
        # occupancy-2 rule below — both W4A8 variants are weight-BW-bound
        # there and 128-row tiles + split-k lose the same way
        auto_tm, auto_tn, auto_sk = _pick_w4_cfg(m_act, n_full, k // 128)
    else:
        auto_tm, auto_tn, auto_sk = _pick_w4a8_cfg(m_act, n_full)
    if tile_m is None:
        tile_m = auto_tm
    if tile_n is None:
        tile_n = auto_tn
    if split_k is None:
        split_k = auto_sk if (tile_m, tile_n) == (auto_tm, auto_tn) else 1
    assert n_full % tile_m == 0, f"padded N ({n_full}) must be divisible by tile_m ({tile_m})"
    if out is None:
        out = torch.empty(m_act, n_full, dtype=torch.bfloat16, device=act.device)
    else:
        assert out.shape == (m_act, n_out) and out.dtype == torch.bfloat16
        assert n_out == n_full, "padded N requires an internally allocated out"
    epi_args = {"v": act_scale.unsqueeze(0)}
    mod = _w4a8_token_scale
    if folded:
        mod = _w4a8_folded_scale
        if chan_scale.shape[0] < n_full:  # N was padded to tile granularity
            chan_scale = torch.cat([chan_scale, chan_scale.new_ones(n_full - chan_scale.shape[0])])
        epi_args["cs"] = chan_scale.to(torch.float32).unsqueeze(0)
    mod.gemm(
        act,
        blob,
        out,
        epi_args=epi_args,
        transform_a=wformat,
        transform_sf=sf,
        tile_M=tile_m,
        tile_N=tile_n,
        tile_K=128,
        cluster_M=1,
        cluster_N=cluster_n,
        max_swizzle_size=max_swizzle_size,
        split_k=split_k,
    )
    if n_out != n_full:
        out = out[:, :n_out]
    return out
