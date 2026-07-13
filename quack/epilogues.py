# Copyright (c) 2026, Han Guo, Tri Dao.
"""Ready-to-use fused-GEMM epilogues, written with the @gemm_epilogue fn
contract (quack.gemm_epilogue) on top of the EpiOp library (quack.epi_ops).

Each entry is a plain function over the accumulator plus declared resources;
kernels are minted/cached per (fn source, op config, tensor metadata). Pass
tensors via ``mod.gemm(A, B, D, C, epi_args={...}, tile_M=..., ...)``.

Sections:
  * packed-polymorphic math helpers (pexp, pabs) — raw cute.math fns are not
    F2/Pair-aware; wrap transcendentals like these do.
  * reusable domain resources live in ``quack.epilogue``: rotary table loads
    in ``rotary`` and per-head RMSNorm statistics in ``head_rmsnorm``.
  * elementwise mods: linear/bias, residual, activation factories, RMS-fused
    (sq-sum reduce), amax (quantization stats), per-tile LSE partials, online
    LSE (stable), transformer-block forward (rms_partial -> rstd_swiglu) and
    the rmsnorm-backward link.
  * paired mods (gated / RoPE) and packed-C/D mods (dgated), via unpack/pack.

Numerics and perf of every mod here are pinned by tests/test_gemm_epilogue.py
(bitwise or 1-ulp vs the hand-written kernels, <=1% perf deltas).
"""

from __future__ import annotations

import cutlass.cute as cute

from quack.activation import (
    dgelu_tanh_approx,
    dswiglu,
    gelu_tanh_approx,
    relu,
    relu_sq,
    swiglu,
)
from quack.epi_ops import ColVecReduce
from quack.epilogue.head_rmsnorm import HeadRMSNormStats
from quack.epilogue.rotary import rotary_cos_sin_load
from quack.gemm_epilogue import F2, gemm_epilogue, pack, unpack


@gemm_epilogue(outputs=("postact",))
def norm_gelu(acc, rstd, weight):
    x = acc * rstd * weight
    return {"D": x, "postact": gelu_tanh_approx(x)}


@gemm_epilogue()
def scaled_residual(acc, c, alpha):
    return {"D": acc * alpha + c}


@gemm_epilogue()
def linear_epi(acc, c, alpha, beta, bias_n, bias_m):
    """The default (linear) epilogue as a mod: alpha*acc + beta*C + rowvec + colvec."""
    return {"D": acc * alpha + c * beta + bias_n + bias_m}


def make_act_mod(fn):
    """Activation-mod factory — exercises closure-salted cache identity."""

    @gemm_epilogue(outputs=("postact",))
    def act_mod(acc):
        return {"D": acc, "postact": fn(acc)}

    return act_mod


relu_mod = make_act_mod(relu)


relu_sq_mod = make_act_mod(relu_sq)


@gemm_epilogue(outputs=("postact",))
def dgelu_mod(acc, c):
    """GemmDAct as a mod: c is the preact, acc is dout; D = dx, postact = act(c)."""
    dx, out = dgelu_tanh_approx(c, acc)
    return {"D": dx, "postact": out}


@gemm_epilogue(outputs=("premult",), reduces={"sqsum": ColVecReduce("sqsum")})
def rms_fused(acc, weight):
    """GemmSqReduce as a mod: sqsum accumulated on pre-scale x, D = x * weight."""
    return {"D": acc * weight, "sqsum": acc * acc, "premult": acc}


@gemm_epilogue(outputs=("postact",), mode="packed_cd_b16x2")
def dswiglu_mod(acc, c):
    """GemmDGated as a mod: acc = dout (per pair), c = packed (x, y) preact.
    Packing is declared by mode and validated against 16-bit C/D at twice GEMM-N."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc)
    return {"D": pack(dx, dy), "postact": out}


@gemm_epilogue(
    outputs=("postact",),
    reduces={"dsum": ColVecReduce("dsum")},
    mode="packed_cd_b16x2",
)
def dswiglu_norm_mod(acc, c, rstd):
    """Full dgated: colvec-scaled dout, reduce on unscaled dout, scaled postact."""
    x, y = unpack(c)
    dx, dy, out = dswiglu(x, y, acc * rstd)
    return {"D": pack(dx, dy), "postact": out * rstd, "dsum": out * acc}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def swiglu_mod(acc):
    """GemmGated as a mod: the accumulator pairs over adjacent N because the
    postact buffer is half of GEMM-N."""
    gate, up = unpack(acc)
    return {"postact": swiglu(gate, up)}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def norm_swiglu_mod(acc, rstd, bias):
    """Gated with rowvec bias (arrives paired) + colvec (scalar), D writeback.
    Pair arithmetic is lane-wise, so the affine part runs before unpacking."""
    v = acc * rstd + bias
    g, u = unpack(v)
    return {"postact": swiglu(g, u), "D": pack(g, u)}


@gemm_epilogue()
def residual_epi(acc, res):
    """Coda Residual: full-tile aux input added to the accumulator."""
    return {"D": acc + res}


@gemm_epilogue(mode="acc_pair")
def rope_epi(acc, table):
    """Coda RoPE: rotate adjacent-N pairs by an interleaved cos/sin table
    ((..., 2j) = cos, (..., 2j+1) = sin), congruent with the D tile."""
    x1, x2 = unpack(acc)
    cos, sin = unpack(table)
    return {"D": pack(x1 * cos - x2 * sin, x1 * sin + x2 * cos)}


def pexp(v):
    """Tuple-polymorphic exp, activation.py-style: raw cute.math fns are not
    F2-aware, so mods needing transcendentals wrap them like this (candidate
    for a shared quack epi-math module)."""
    if isinstance(v, tuple):
        return F2(*cute.arch.exp_packed_f32x2(v))
    return cute.math.exp(v, fastmath=True)


@gemm_epilogue(reduces={"sexp": ColVecReduce("sexp")})
def lse_partial_epi(acc, scale):
    """Coda LSE, per-tile flavor: sexp[m, tile] = sum_n exp(acc * scale);
    the host finalizes log(sum(partials)). NOTE: no online max — needs a
    max-combine reduce for large-logit stability (Coda's LSEReduce is online)."""
    return {"D": acc, "sexp": pexp(acc * scale)}


def pabs(v):
    if isinstance(v, tuple):
        return F2(pabs(v[0]), pabs(v[1]))
    return cute.arch.fmax(v, -v)


@gemm_epilogue(reduces={"amax": ColVecReduce("amax", combine="max")})
def amax_epi(acc):
    """Per-tile column amax — the quantized-output (SFD) building block.
    |x| >= 0, so the zero OOB accumulator lanes of a ragged last tile can't
    corrupt the max (see VecReduce.combine note)."""
    return {"D": acc, "amax": pabs(acc)}


def _sq_prepass(acc):
    """Prepass fn: the statistic input, explicit (this replaces epirope's
    _prenorm_vec_ops replay registry — any pre-norm transform would be
    duplicated here in plain sight)."""
    return {"qk": acc * acc}


@gemm_epilogue(
    ops={"qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
)
def qknorm_epi(acc, qk):
    return {"D": acc * qk}


@gemm_epilogue(
    ops={"cs": rotary_cos_sin_load("cs"), "qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def qk_rope_epi(acc, cs, qk):
    """The full epirope composition: per-head RMSNorm (prepass stats) then
    rotary, in five lines of fn math. TMA table (see rotary_cos_sin_load):
    at the winning clustered-pingpong configs the LDG table's register cost
    is what tips this composition into spills."""
    x1, x2 = unpack(acc * qk)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(
    ops={"cs": rotary_cos_sin_load("cs", tma=False), "qk": HeadRMSNormStats("qk", eps=1e-6)},
    prepass=_sq_prepass,
    prepass_outs=("qk",),
    mode="acc_pair",
)
def qk_rope_ldg_epi(acc, cs, qk):
    """qk_rope_epi on the gmem->rmem table op (see rope_table_ldg_epi)."""
    x1, x2 = unpack(acc * qk)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(outputs=("resid_out",), reduces={"sqsum": ColVecReduce("sqsum")})
def rms_partial_epi(acc, c, weight):
    """GEMM1 of a block: y = acc + residual(C); write the residual stream (aux),
    the weight-applied output (D — rstd deferred: row scaling commutes through
    the NEXT gemm), and the per-tile sq-sum partials for rstd finalization."""
    y = acc + c
    return {"D": y * weight, "resid_out": y, "sqsum": y * y}


@gemm_epilogue(outputs=("postact",), mode="acc_pair")
def rstd_swiglu_epi(acc, rstd):
    """GEMM2 of a block: apply the deferred rstd (colvec), then swiglu pairs."""
    g, u = unpack(acc * rstd)
    return {"postact": swiglu(g, u)}


@gemm_epilogue(reduces={"dots": ColVecReduce("dots")})
def rms_bwd_partial_epi(acc, y, rstd, w):
    """RMSNorm backward around a dgrad GEMM: acc = dz @ W2^T (= d(norm out)).
    t = acc*w, xhat = saved_prenorm(TileLoad) * rstd; write D = rstd*t and the
    per-tile partials of the correction dot mean(t * xhat)."""
    t = acc * w
    xhat = y * rstd
    return {"D": t * rstd, "dots": t * xhat}
