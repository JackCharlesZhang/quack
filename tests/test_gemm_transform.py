# Copyright (c) 2026, Tri Dao.
"""Gates for the RS mainloop (mma_is_rs=True) and the A-operand transform
frontend riding it.

RS gate: bitwise-identical to the SS mainloop — same WGMMA instruction, same
k-tile order, same accumulation order; only the A operand source differs
(ldmatrix s2r load of the fragment vs the SS descriptor read).

Value-fn gate: an ``@a_transform`` fn applied on the fragment must be bitwise
== pre-applying it to A on the host (exact for powers of 2) and running the
plain SS mainloop."""

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


def _run_gemm(A, B, D, tile_mnk, cluster_mnk, pingpong, mma_is_rs, transform_a=None, aux=None):
    from quack.operand_transform import TransformAOperand

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    mA = from_dlpack(A, assumed_align=16)
    if aux is not None:
        mA = TransformAOperand(mA, from_dlpack(aux, assumed_align=16))
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
        transform_a=transform_a,
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


# ── @a_transform fn frontend (value family) ──────────────────────────────────

from quack.operand_transform import a_transform  # noqa: E402


@a_transform(vec_size=2)
def _identity2_a(x):
    return x


@a_transform(vec_size=8)
def _identity8_a(x):
    return x


@a_transform(vec_size=4)
def _halve_a(x):
    return x * 0.5


@a_transform(vec_size=2, consts=lambda: 0.5)
def _scale_by_const_a(x, c):
    return x * c


def _check_value_mod(mod, ref_scale, pingpong=False, a_major="k", tile_mnk=(128, 128, 64)):
    """Value-fn gate: fn on the fragment must be bitwise == pre-scaling A on
    the host (exact for powers of 2) and running the plain SS mainloop."""
    torch.manual_seed(0)
    m, n, k = 384, 256, 288
    if a_major == "k":
        A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    else:
        A = (torch.randn(k, m, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)).t()
    B = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    A_ref = (A.float() * ref_scale).to(torch.bfloat16)
    D_ss = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    D_fn = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _run_gemm(
        A_ref.contiguous() if a_major == "k" else A_ref,
        B,
        D_ss,
        tile_mnk,
        (1, 1, 1),
        pingpong,
        mma_is_rs=False,
    )
    _run_gemm(A, B, D_fn, tile_mnk, (1, 1, 1), pingpong, mma_is_rs=False, transform_a=mod)
    torch.cuda.synchronize()
    assert torch.equal(D_fn, D_ss), "value fn is not bitwise vs host-prescaled SS"


@pytest.mark.parametrize("mod", [_identity2_a, _identity8_a], ids=["vec2", "vec8"])
def test_a_transform_identity(mod):
    _check_value_mod(mod, ref_scale=1.0)


@pytest.mark.parametrize("a_major", ["k", "m"])
def test_a_transform_halve(a_major):
    _check_value_mod(_halve_a, ref_scale=0.5, a_major=a_major)


def test_a_transform_halve_pingpong():
    _check_value_mod(_halve_a, ref_scale=0.5, pingpong=True)


def test_a_transform_consts():
    # consts() is hoisted once per kernel and arrives as the fn's last arg
    _check_value_mod(_scale_by_const_a, ref_scale=0.5)


def test_a_transform_semantic_keys():
    assert _identity2_a.__quack_semantic_key__() != _identity8_a.__quack_semantic_key__()
    assert _identity2_a.__quack_semantic_key__() != _halve_a.__quack_semantic_key__()

    def make_scale(c):
        @a_transform(vec_size=2)
        def scale(x):
            return x * c

        return scale

    assert make_scale(0.5).__quack_semantic_key__() != make_scale(0.25).__quack_semantic_key__(), (
        "closure constants must reach the semantic key"
    )


# ── colvec_ktile: per-(A-row, k-tile) runtime operand via the aux TMA slot ───
#
# The linear-CE dx shape: multiply the A fragment by a per-(row, k-tile)
# power-of-two scale (exact in bf16), strip delivered per k-tile under the AB
# mbarrier. Gate: bitwise == pre-scaling A on the host and running plain SS.


@a_transform(vec_size=8, args={"u": "colvec_ktile"})
def _colvec_ktile_scale_a(x, u):
    return x * u


def _make_colvec_strip_case(m, n, k, tile_k, gran=None, a_major="k", seed=3):
    torch.manual_seed(seed)
    gran = tile_k if gran is None else gran
    if a_major == "k":
        A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    else:
        A = (torch.randn(k, m, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)).t()
    B = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    # one row per k-group of gran elements, K padded to whole k-tiles
    kg = -(-k // tile_k) * (tile_k // gran)
    # power-of-two scales: the bf16 multiply is exact on both sides
    strip = torch.ldexp(
        torch.ones(kg, m, device="cuda"), torch.randint(-3, 4, (kg, m), device="cuda")
    ).to(torch.bfloat16)
    ks = torch.arange(k, device="cuda") // gran
    A_pre = (A.float() * strip.float().t()[:, ks]).to(torch.bfloat16)
    return A, B, strip, A_pre


def _check_colvec_strip(
    m=384,
    n=256,
    k=288,
    tile_mnk=(128, 128, 64),
    cluster_mnk=(1, 1, 1),
    pingpong=False,
    a_major="k",
    mod=None,
    gran=None,
):
    from quack.operand_transform.host import transform_a_operand

    mod = mod if mod is not None else _colvec_ktile_scale_a
    A, B, strip, A_pre = _make_colvec_strip_case(m, n, k, tile_mnk[2], gran, a_major)
    aux = transform_a_operand(mod, A, {"u": strip}, tile_mnk[0], tile_mnk[2]).sf
    D_ss = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    D_fn = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _run_gemm(A_pre.contiguous(), B, D_ss, tile_mnk, cluster_mnk, pingpong, mma_is_rs=False)
    _run_gemm(
        A,
        B,
        D_fn,
        tile_mnk,
        cluster_mnk,
        pingpong,
        mma_is_rs=False,
        transform_a=mod,
        aux=aux,
    )
    torch.cuda.synchronize()
    # Sanity: a correct GEMM at all.
    ref = (A_pre.float() @ B.float().mT).to(torch.bfloat16)
    torch.testing.assert_close(D_fn, ref, atol=3e-2, rtol=1e-3)
    # The gate: exact multiply + identical accumulation order => bitwise.
    assert torch.equal(D_fn, D_ss), "colvec strip fn is not bitwise vs host-prescaled SS"


@pytest.mark.parametrize("a_major", ["k", "m"])
def test_a_transform_colvec_ktile(a_major):
    # k = 4.5 k-tiles exercises the ragged operand tail with the TMA zero-fill
    _check_colvec_strip(a_major=a_major)


def test_a_transform_colvec_ktile_pingpong():
    _check_colvec_strip(pingpong=True)


def test_a_transform_colvec_ktile_cluster():
    # cluster_N = 2 exercises the aux multicast opt-out (dup box loads)
    _check_colvec_strip(m=512, n=512, k=256, cluster_mnk=(1, 2, 1))


@pytest.mark.parametrize(
    "tile_mnk",
    [
        (64, 128, 64),  # atom (1, 1): single MMA warpgroup, tm64 = 1
        (256, 128, 64),  # atom (2, 1): tm64 = 4 aux box
        (192, 256, 64),  # atom (1, 2): N-split warpgroups
    ],
)
def test_a_transform_colvec_ktile_atom_layouts(tile_mnk):
    _check_colvec_strip(m=768, n=512, k=256, tile_mnk=tile_mnk)


@a_transform(vec_size=8, args={"u": "kvec_m64"})
def _kvec_m64_scale_a(x, u):
    return x * u


def _check_kvec_m64(
    m=384, n=256, k=288, tile_mnk=(128, 128, 64), cluster_mnk=(1, 1, 1), a_major="k"
):
    """kvec_m64 gate (the LCE dw strip shape): one pow2 value per (m64 row
    block, k-element), bitwise vs host-prescaled SS. The strip is (M/64,
    rk*tile_K) k-contiguous, K padded to whole k-tiles."""
    from quack.operand_transform.host import transform_a_operand

    torch.manual_seed(5)
    if a_major == "k":
        A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    else:
        A = (torch.randn(k, m, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)).t()
    B = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / math.sqrt(k)
    kg = -(-k // tile_mnk[2]) * tile_mnk[2]
    strip = torch.ldexp(
        torch.ones(m // 64, kg, device="cuda"), torch.randint(-3, 4, (m // 64, kg), device="cuda")
    ).to(torch.bfloat16)
    rows = torch.arange(m, device="cuda") // 64
    A_pre = (A.float() * strip.float()[rows, :k]).to(torch.bfloat16)
    aux = transform_a_operand(_kvec_m64_scale_a, A, {"u": strip}, tile_mnk[0], tile_mnk[2]).sf
    D_ss = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    D_fn = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    _run_gemm(A_pre.contiguous(), B, D_ss, tile_mnk, cluster_mnk, False, mma_is_rs=False)
    _run_gemm(
        A,
        B,
        D_fn,
        tile_mnk,
        cluster_mnk,
        False,
        mma_is_rs=False,
        transform_a=_kvec_m64_scale_a,
        aux=aux,
    )
    torch.cuda.synchronize()
    ref = (A_pre.float() @ B.float().mT).to(torch.bfloat16)
    torch.testing.assert_close(D_fn, ref, atol=3e-2, rtol=1e-3)
    assert torch.equal(D_fn, D_ss), "kvec_m64 fn is not bitwise vs host-prescaled SS"


@pytest.mark.parametrize("a_major", ["k", "m"])
def test_a_transform_kvec_m64(a_major):
    # m-major A is the dw orientation (A = E^T); k tail = 4.5 k-tiles
    _check_kvec_m64(a_major=a_major)


@pytest.mark.parametrize("tile_mnk", [(256, 128, 64), (192, 256, 64)], ids=["atom21", "atom12"])
def test_a_transform_kvec_m64_atom_layouts(tile_mnk):
    _check_kvec_m64(m=768, n=512, k=256, tile_mnk=tile_mnk)


def test_a_transform_kvec_m64_cluster():
    _check_kvec_m64(m=512, n=512, k=256, cluster_mnk=(1, 2, 1))


@a_transform(vec_size=8, args={"u": "colvec_k16"})
def _colvec_k16_scale_a(x, u):
    return x * u


@a_transform(vec_size=8, args={"u": "colvec_k32"})
def _colvec_k32_scale_a(x, u):
    return x * u


@pytest.mark.parametrize(
    "mod,gran", [(_colvec_k16_scale_a, 16), (_colvec_k32_scale_a, 32)], ids=["k16", "k32"]
)
def test_a_transform_colvec_strip_granularity(mod, gran):
    # dense blockscaled-SF granularities: g = tile_K/gran values per row per
    # k-tile; the ragged K tail (4.5 k-tiles) pads the strip to whole tiles
    _check_colvec_strip(mod=mod, gran=gran)
    _check_colvec_strip(m=768, n=512, k=256, tile_mnk=(256, 128, 64), mod=mod, gran=gran)


def test_a_transform_colvec_ktile_via_host_plan():
    """Runtime-operand transforms flow through the generic host layer: A
    crosses as the transform_a_operand bundle, the operand fake is derived
    from the mod + tile_M, and the launch is bitwise == the plain plan on
    prescaled A."""
    from quack.gemm_host import build_gemm_epi_plan, run_gemm_epi_plan
    from quack.operand_transform.host import transform_a_operand

    m, n, k = 384, 256, 512
    A, B, strip, A_pre = _make_colvec_strip_case(m, n, k, tile_k=64, seed=11)
    bundle = transform_a_operand(_colvec_ktile_scale_a, A, {"u": strip}, 128, 64)
    cap = get_device_capacity(A.device)

    def make_plan(a_arg, transform):
        return build_gemm_epi_plan(
            GemmDefaultSm90,
            cap,
            a_arg,
            B,
            torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
            None,
            epi_values={},
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            transform_a=transform,
        )

    D_t = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    run_gemm_epi_plan(make_plan(bundle, _colvec_ktile_scale_a), bundle, B, D_t, None, {})
    D_ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    run_gemm_epi_plan(make_plan(A_pre, None), A_pre.contiguous(), B, D_ref, None, {})
    torch.cuda.synchronize()
    assert torch.equal(D_t, D_ref)


def test_a_transform_args_validation():
    with pytest.raises(AssertionError, match="unknown operand kind"):

        @a_transform(vec_size=8, args={"u": "rowvec_ktile"})
        def _bad_kind(x, u):
            return x

    with pytest.raises(AssertionError, match="not parameters"):

        @a_transform(vec_size=8, args={"v": "colvec_ktile"})
        def _bad_name(x, u):
            return x

    # the operand declaration reaches the semantic key
    @a_transform(vec_size=8)
    def _no_args(x, u):
        return x

    assert (
        _colvec_ktile_scale_a.__quack_semantic_key__()[-1] != _no_args.__quack_semantic_key__()[-1]
    )


def test_a_transform_via_host_plan():
    """Value transforms are a first-class axis of the generic host layer:
    build_gemm_epi_plan(transform_a=<mod>) compiles/launches through the same
    machinery as every epilogue variant, bitwise == the plain plan on
    host-prescaled A (exact for the power-of-2 fn)."""
    from quack.gemm_host import build_gemm_epi_plan, run_gemm_epi_plan

    torch.manual_seed(7)
    m, n, k = 256, 256, 512
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    cap = get_device_capacity(A.device)

    def make_plan(transform):
        return build_gemm_epi_plan(
            GemmDefaultSm90,
            cap,
            A,
            B,
            torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
            None,
            epi_values={},
            tile_M=128,
            tile_N=128,
            cluster_M=1,
            cluster_N=1,
            transform_a=transform,
        )

    D_t = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    run_gemm_epi_plan(make_plan(_halve_a), A, B, D_t, None, {})
    D_ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    run_gemm_epi_plan(make_plan(None), (A * 0.5).contiguous(), B, D_ref, None, {})
    torch.cuda.synchronize()
    assert torch.equal(D_t, D_ref)


def test_a_transform_epi_mod_composition():
    """Value transforms compose with @gemm_epilogue fns through the eager
    mod(A, B, transform_a=...) surface (autotune bypassed; default config)."""
    from quack.gemm_epilogue import gemm_epilogue

    @gemm_epilogue()
    def _plus_bias(acc, bias):
        return {"D": acc + bias}

    torch.manual_seed(9)
    m, n, k = 256, 512, 512
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    Bkn = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)  # (k, n) logical
    bias = torch.randn(1, n, device="cuda", dtype=torch.float32)
    out = _plus_bias(A, Bkn, transform_a=_halve_a, bias=bias)["D"]
    ref = (A.float() * 0.5) @ Bkn.float() + bias
    atol = ref.abs().max().item() * 2**-7 + 1e-5
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=1e-2)
