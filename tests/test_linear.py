# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch
import torch.nn.functional as F

from quack.linear import linear_func, linear_act_func
from quack.gemm_interface import (
    gemm,
    gemm_ref,
    gemm_add,
    gemm_add_ref,
    gemm_add_inplace,
    gemm_dact,
    gemm_act_ref,
    gemm_dact_ref,
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear(in_features, out_features, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype, requires_grad=True)
    x = x[::2]  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    out = linear_func(x, w, tuned=False)  # Disable tuning for faster test
    out_ref = F.linear(x.float(), w.float())
    out_pt = F.linear(x, w)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
    dout = torch.randn_like(out)
    dx, dw = torch.autograd.grad(out, (x, w), dout)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x, w), dout)
    dx_pt, dw_pt = torch.autograd.grad(out_pt, (x, w), dout)
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-6
    assert (dw - dw_ref).abs().max() < 2 * (dw_pt - dw_ref).abs().max() + 1e-6


@pytest.mark.parametrize("store_preact", [False, True])
@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear_act(in_features, out_features, input_dtype, activation, store_preact):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype, requires_grad=True)
    x = x[::2]  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    # Disable tuning for faster test
    preact, postact = linear_act_func(x, w, activation, store_preact=store_preact, tuned=False)
    preact_ref, postact_ref = gemm_act_ref(
        x.float(), w.float().T, activation=activation, store_preact=store_preact
    )
    preact_pt, postact_pt = gemm_act_ref(x, w.T, activation=activation, store_preact=store_preact)
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-6
    if store_preact:
        assert preact is not None and preact_ref is not None
        assert (preact - preact_ref).abs().max() < 2 * (preact_pt - preact_ref).abs().max() + 1e-6


@pytest.mark.parametrize("activation", ["relu", "relu_sq", "gelu_tanh_approx"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("n", [1504, 2048])
def test_gemm_dact(n, k, input_dtype, activation):
    """Test GEMM with activation gradient computation."""
    device = "cuda"
    torch.random.manual_seed(0)
    m = 960
    dout_input = torch.randn((m, k), device=device, dtype=input_dtype)
    weight = torch.randn((n, k), device=device, dtype=input_dtype) / math.sqrt(k)
    preact = torch.randn((m, n), device=device, dtype=input_dtype, requires_grad=True)
    # Disable tuning for faster test
    dx, postact = gemm_dact(dout_input, weight.T, preact, activation=activation, tuned=False)
    dx_ref, postact_ref = gemm_dact_ref(
        dout_input.float(), weight.float().T, preact.float(), activation=activation
    )
    dx_pt, postact_pt = gemm_dact_ref(dout_input, weight.T, preact, activation=activation)
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("m", [960, 1920])
def test_gemm_add_inplace(m, k, n, input_dtype):
    """Test in-place GEMM with addition: C += A @ B."""
    device = "cuda"
    torch.random.manual_seed(0)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    # Save original C for reference computation
    C_og = C.clone()
    gemm_add_inplace(A, B, C, tuned=False)
    C_ref = C_og.float() + torch.mm(A.float(), B.float())
    C_pt = C_og + torch.mm(A, B)
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-5


@pytest.mark.parametrize("alpha_beta_type", ["float", "tensor"])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("k", [256, 768])
@pytest.mark.parametrize("m", [480, 960])
def test_gemm_add_inplace_alpha_beta(m, k, n, input_dtype, alpha, beta, alpha_beta_type):
    """Test in-place GEMM with alpha/beta scaling: C = alpha * A @ B + beta * C."""
    device = "cuda"
    torch.random.manual_seed(42)
    A = torch.randn((m, k), device=device, dtype=input_dtype)
    B = torch.randn((k, n), device=device, dtype=input_dtype)
    C = torch.randn((m, n), device=device, dtype=input_dtype)
    if alpha_beta_type == "tensor":
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    C_og = C.clone()
    gemm_add_inplace(A, B, C, alpha=alpha, beta=beta, tuned=False)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    C_ref = alpha_val * torch.mm(A.float(), B.float()) + beta_val * C_og.float()
    C_pt = alpha_val * torch.mm(A, B) + beta_val * C_og
    assert (C - C_ref).abs().max() < 2 * (C_pt - C_ref).abs().max() + 1e-4


@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("alpha", [1.0, 1.7])
# @pytest.mark.parametrize("alpha_is_tensor", [False])
# @pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("B_major", ["k", "n"])
# @pytest.mark.parametrize("B_major", ["k"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
# @pytest.mark.parametrize("n", [2048])
# @pytest.mark.parametrize("k", [736])
@pytest.mark.parametrize("num_groups", [3, 5])
# @pytest.mark.parametrize("num_groups", [3])
def test_gemm_varlen_m(
    num_groups, k, n, input_dtype, B_major, dynamic_scheduler, alpha, alpha_is_tensor
):
    """Test GEMM with variable length M dimension using cu_seqlens_m."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(100, 500, (num_groups,), device="cpu")
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_m = cu_seqlens_m.to(device)
    A = torch.randn((total_m, k), device=device, dtype=input_dtype)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    out = gemm(
        A,
        B,
        alpha=alpha,
        cu_seqlens_m=cu_seqlens_m,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    out_ref = gemm_ref(A.float(), B.float(), alpha=alpha, cu_seqlens_m=cu_seqlens_m)
    out_pt = gemm_ref(A, B, alpha=alpha, cu_seqlens_m=cu_seqlens_m)
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("alpha", [1.0, 1.5])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1504, 2048])
@pytest.mark.parametrize("k", [736, 1024])
@pytest.mark.parametrize("num_groups", [3, 5])
def test_gemm_add_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
):
    """Test GEMM with addition and variable length M dimension using cu_seqlens_m."""
    device = "cuda"
    torch.random.manual_seed(0)
    seq_lens = torch.randint(100, 500, (num_groups,), device="cpu")
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_m = cu_seqlens_m.to(device)
    A = torch.randn((total_m, k), device=device, dtype=input_dtype)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    C = torch.randn((total_m, n), device=device, dtype=input_dtype)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    out = gemm_add(
        A,
        B,
        C,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out_ref = gemm_add_ref(
        A.float(), B.float(), C.float(), alpha=alpha_val, beta=beta_val, cu_seqlens_m=cu_seqlens_m
    )
    out_pt = gemm_add_ref(A, B, C, alpha=alpha_val, beta=beta_val, cu_seqlens_m=cu_seqlens_m)
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-5


@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("alpha", [1.0, 2.0])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
@pytest.mark.parametrize("B_major", ["k", "n"])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("k", [512, 768])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_inplace_varlen_m(
    num_groups,
    k,
    n,
    input_dtype,
    B_major,
    dynamic_scheduler,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
):
    """Test in-place GEMM with addition and variable length M dimension: out = alpha * A @ B + beta * out."""
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_m = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_m = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_m = cu_seqlens_m.to(device)
    A = torch.randn((total_m, k), device=device, dtype=input_dtype)
    B = torch.randn((num_groups, k, n), device=device, dtype=input_dtype) / math.sqrt(k)
    out = torch.randn((total_m, n), device=device, dtype=input_dtype)
    if B_major == "k":
        B = B.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    # Save original out for reference computation
    out_og = out.clone()
    gemm_add_inplace(
        A,
        B,
        out,
        alpha=alpha,
        beta=beta,
        cu_seqlens_m=cu_seqlens_m,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out_ref = gemm_add_ref(
        A.float(),
        B.float(),
        out_og.float(),
        out=None,  # Don't use in-place for reference
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_m=cu_seqlens_m,
    )
    out_pt = gemm_add_ref(
        A, B, out_og, out=None, alpha=alpha_val, beta=beta_val, cu_seqlens_m=cu_seqlens_m
    )
    assert out.shape == (total_m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({total_m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4
