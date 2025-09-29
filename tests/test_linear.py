# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch
import torch.nn.functional as F

from quack.linear import linear_func, linear_act_func
from quack.gemm_interface import (
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

 
    print(out[:20])
    print(out_ref[:20])
    print(out.size(), out_ref.size())

    print("max diff", (out - out_ref).abs().max())
    print("sum of out:", out.sum())
    print("sum of out_ref:", out_ref.sum())
    print("sum diff:", (out.sum() - out_ref.sum()).abs())

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
