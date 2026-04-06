import pytest
import torch

import cutlass

from quack.blockscaled_gemm_utils import (
    blockscaled_gemm_reference,
    compile_blockscaled_gemm_tvm_ffi,
    create_blockscaled_operand_quantized,
    create_blockscaled_operand_tensor,
    create_blockscaled_scale_tensor,
    scale_blocked_for_cublas,
    scale_view_for_kernel,
)
from quack.gemm_default_epi import GemmDefaultSm100
from quack.mx_utils import to_blocked


def _skip_if_not_sm100():
    major = torch.cuda.get_device_properties(0).major
    if major < 10:
        pytest.skip("SM100+ required")


def _compile_blockscaled_gemm(
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    m,
    n,
    k,
    l,
):
    a_ref, mA = create_blockscaled_operand_tensor(l, m, k, False, ab_dtype)
    b_ref, mB = create_blockscaled_operand_tensor(l, n, k, False, ab_dtype)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, d_dtype, init="empty")
    sfa_ref, mSFA = create_blockscaled_scale_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, mSFB = create_blockscaled_scale_tensor(l, n, k, sf_vec_size, sf_dtype)
    compiled = compile_blockscaled_gemm_tvm_ffi(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
    )
    return (
        compiled,
        (mA, mB, mD, mSFA, mSFB),
        (a_ref, b_ref, sfa_ref, sfb_ref, mD),
    )


def _run_blockscaled_gemm(compiled, args):
    mA, mB, mD, mSFA, mSFB = args
    compiled(mA, mB, mD, mSFA, mSFB)
    torch.cuda.synchronize()


def test_blockscaled_validation():
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 64),
        (1, 1),
        256,
        64,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 192),
        (1, 1),
        256,
        192,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.Float32,
        (128, 192),
        (1, 1),
        256,
        192,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (256, 384),
        (2, 1),
        256,
        512,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (256, 224),
        (2, 1),
        256,
        448,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        (256, 384),
        (2, 1),
        256,
        512,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (64, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        32,
        cutlass.Float32,
        (128, 128),
        (1, 1),
        256,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )
    assert not GemmDefaultSm100.can_implement_blockscaled(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        32,
        cutlass.BFloat16,
        (256, 128),
        (1, 1),
        512,
        256,
        256,
        1,
        "k",
        "k",
        "n",
    )


@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,sf_vec_size,d_dtype,mma_tiler_mn,cluster_shape_mn,m,n,k,l",
    [
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 64),
            (1, 1),
            256,
            64,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 192),
            (1, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (128, 128),
            (1, 1),
            256,
            256,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 64),
            (2, 1),
            512,
            64,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 192),
            (2, 1),
            512,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.BFloat16,
            (256, 128),
            (2, 1),
            512,
            256,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 192),
            (2, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (128, 128),
            (1, 1),
            256,
            256,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E8M0FNU,
            16,
            cutlass.Float32,
            (128, 64),
            (1, 1),
            256,
            64,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (256, 192),
            (2, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (128, 192),
            (1, 1),
            256,
            192,
            256,
            1,
        ),
        (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.Float32,
            (256, 224),
            (2, 1),
            256,
            224,
            256,
            1,
        ),
    ],
)
def test_blockscaled_correctness(
    ab_dtype, sf_dtype, sf_vec_size, d_dtype, mma_tiler_mn, cluster_shape_mn, m, n, k, l
):
    _skip_if_not_sm100()

    (
        compiled,
        args,
        (a_ref, b_ref, sfa_ref, sfb_ref, _),
    ) = _compile_blockscaled_gemm(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        d_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
    )
    _run_blockscaled_gemm(compiled, args)

    _, _, d_torch, _, _ = args
    ref = blockscaled_gemm_reference(a_ref, b_ref, sfa_ref, sfb_ref)
    err = (d_torch.float() - ref).abs().max().item()
    tol = 5e-3 if d_dtype != cutlass.Float32 else 5e-4
    assert err < tol, f"max_err={err}"


# ---------------------------------------------------------------------------
# Scale layout invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mn,sf_k,l", [(128, 4, 1), (256, 16, 1), (384, 12, 2), (512, 8, 1)])
def test_scale_layout_matches_cublas(mn, sf_k, l):
    """The quack kernel scale-view and cuBLAS's to_blocked must share the
    same underlying byte layout (they both represent the PTX
    tcgen05 scale-factor atom, tiled in the same outer order)."""
    torch.manual_seed(0)
    # a 2D uint8 scale slice per batch
    scale_2d = torch.randint(0, 255, (l, mn, sf_k), device="cuda", dtype=torch.uint8)

    # Build our contiguous scale storage via create_blockscaled_operand_quantized's
    # rearrangement logic: pad + (l, rm, 128, rk, 4) -> (l, rm, rk, 32, 4, 4)
    rm = (mn + 127) // 128
    rk = (sf_k + 3) // 4
    mn_pad = rm * 128
    sf_k_pad = rk * 4
    padded = torch.zeros(l, mn_pad, sf_k_pad, device="cuda", dtype=torch.uint8)
    padded[:, :mn, :sf_k] = scale_2d
    blocks = padded.view(l, rm, 128, rk, 4).permute(0, 1, 3, 2, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, 4).transpose(3, 4).contiguous()
    scale_contig = blocks  # (l, rm, rk, 32, 4, 4)

    # kernel view indexing must map [m%32, (m//32)%4, m//128, k%4, k//4, l] -> scale_2d[l, m, k]
    kv = scale_view_for_kernel(scale_contig.view(torch.float8_e8m0fnu), mn, sf_k, l).view(
        torch.uint8
    )
    m_positions = sorted({0, 1, 17, 31, 33, 127, min(128, mn - 1), mn - 1} & set(range(mn)))
    k_positions = sorted({0, 1, 3, 4, 7, sf_k - 1} & set(range(sf_k)))
    for li in range(l):
        for mi in m_positions:
            for ki in k_positions:
                assert (
                    kv[mi % 32, (mi // 32) % 4, mi // 128, ki % 4, ki // 4, li].item()
                    == scale_2d[li, mi, ki].item()
                ), f"mismatch at l={li} m={mi} k={ki}"

    # cuBLAS slice must equal to_blocked(scale_2d[l])
    for li in range(l):
        ours = scale_blocked_for_cublas(scale_contig.view(torch.float8_e8m0fnu), mn, sf_k, li).view(
            torch.uint8
        )
        ref = to_blocked(scale_2d[li])
        assert torch.equal(ours, ref), f"to_blocked mismatch at l={li}"


# ---------------------------------------------------------------------------
# End-to-end: quantized MXFP8 inputs through quack kernel vs cuBLAS vs dequant ref
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mma_tiler_mn,cluster_shape_mn,m,n,k",
    [
        # All 5 supported blockscaled tile_n values (64, 128, 192, 224, 256).
        ((128, 64), (1, 1), 256, 64, 512),
        ((128, 128), (1, 1), 256, 256, 256),
        ((128, 128), (1, 1), 512, 512, 512),
        ((128, 192), (1, 1), 256, 192, 256),
        ((128, 256), (1, 1), 256, 256, 256),
        ((256, 128), (2, 1), 512, 256, 512),
        ((256, 192), (2, 1), 256, 192, 256),
        ((256, 192), (2, 1), 256, 384, 256),
        ((256, 192), (2, 1), 512, 192, 512),
        ((256, 224), (2, 1), 256, 224, 256),
        ((256, 224), (2, 1), 512, 224, 512),
        ((256, 256), (2, 1), 512, 256, 512),
    ],
)
def test_blockscaled_mxfp8_quantized(mma_tiler_mn, cluster_shape_mn, m, n, k):
    _skip_if_not_sm100()
    l, sf_vec = 1, 32

    torch.manual_seed(0)
    a_ref, mA, a_sc = create_blockscaled_operand_quantized(l, m, k, False, sf_vec)
    b_ref, mB, b_sc = create_blockscaled_operand_quantized(l, n, k, False, sf_vec)
    _, mD = create_blockscaled_operand_tensor(l, m, n, False, cutlass.BFloat16, init="empty")

    mSFA = scale_view_for_kernel(a_sc, m, k // sf_vec, l)
    mSFB = scale_view_for_kernel(b_sc, n, k // sf_vec, l)

    runner = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float8E4M3FN,
        cutlass.Float8E8M0FNU,
        sf_vec,
        cutlass.BFloat16,
        mma_tiler_mn,
        cluster_shape_mn,
        mA,
        mB,
        mD,
        mSFA,
        mSFB,
    )
    runner(mA, mB, mD, mSFA, mSFB)
    torch.cuda.synchronize()

    # Reference: dequant matmul (a_ref/b_ref are already dequantized)
    d_ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    err = (mD.float() - d_ref).abs().max().item()
    assert err < 5e-3, f"quack vs dequant max_err={err}"

    # cuBLAS: bit-exact match expected (same operand bits, same scale bytes, same hw MMA)
    a_cub = mA[:, :, 0].contiguous()
    b_cub = mB[:, :, 0].contiguous()
    a_sc_cub = scale_blocked_for_cublas(a_sc, m, k // sf_vec, 0)
    b_sc_cub = scale_blocked_for_cublas(b_sc, n, k // sf_vec, 0)
    out_cublas = torch._scaled_mm(
        a_cub,
        b_cub.t(),
        scale_a=a_sc_cub,
        scale_b=b_sc_cub,
        out_dtype=torch.bfloat16,
    )
    assert torch.equal(mD.squeeze(-1), out_cublas), (
        f"quack != cuBLAS: max_err={(mD.squeeze(-1).float() - out_cublas.float()).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# High-level PyTorch interface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape_mnk", [(256, 256, 256), (512, 256, 256), (128, 128, 256)])
@pytest.mark.parametrize("batched", [False, True])
def test_mxfp8_interface(shape_mnk, batched):
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import (
        mxfp8_gemm,
        mxfp8_gemm_cublas,
        mxfp8_gemm_ref,
        mxfp8_gemm_quantize,
        mxfp8_quantize,
    )

    M, N, K = shape_mnk
    L = 2 if batched else 1
    torch.manual_seed(0)
    shape_A = (L, M, K) if batched else (M, K)
    # Weight stored nn.Linear-style (N, K) row-major; pass .mT to get K-contig (K, N)
    shape_W = (L, N, K) if batched else (N, K)
    A_hp = torch.randn(*shape_A, device="cuda", dtype=torch.bfloat16) * K**-0.5
    W_hp = torch.randn(*shape_W, device="cuda", dtype=torch.bfloat16) * K**-0.5

    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)  # (..., N, K), (..., N, K/32)
    assert A_q.dtype == torch.float8_e4m3fn and A_sc.dtype == torch.float8_e8m0fnu

    B_q = W_q.mT  # (..., K, N) K-contig view
    B_sc = W_sc.mT  # (..., K/32, N) K-contig view

    out = mxfp8_gemm(A_q, B_q, A_sc, B_sc)
    assert out.shape == ((L, M, N) if batched else (M, N))
    assert out.dtype == torch.bfloat16

    ref = mxfp8_gemm_ref(A_q, B_q, A_sc, B_sc)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < 5e-3, f"quack vs ref max_err={err}"

    # cuBLAS comparison only for 2D / L=1
    if not batched:
        out_cublas = mxfp8_gemm_cublas(A_q, B_q, A_sc, B_sc)
        assert torch.equal(out, out_cublas), "quack interface != cuBLAS"

    # High-level quantize+gemm convenience fn
    out2 = mxfp8_gemm_quantize(A_hp, W_hp)
    assert torch.equal(out, out2)


def test_mxfp8_interface_preallocated_out():
    _skip_if_not_sm100()
    from quack.gemm_blockscaled_interface import mxfp8_gemm, mxfp8_quantize

    M, N, K = 256, 256, 256
    torch.manual_seed(0)
    A_hp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * K**-0.5
    W_hp = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * K**-0.5
    A_q, A_sc = mxfp8_quantize(A_hp)
    W_q, W_sc = mxfp8_quantize(W_hp)
    B_q, B_sc = W_q.mT, W_sc.mT

    out_alloc = mxfp8_gemm(A_q, B_q, A_sc, B_sc)
    out_pre = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    mxfp8_gemm(A_q, B_q, A_sc, B_sc, out=out_pre)
    assert torch.equal(out_alloc, out_pre)
