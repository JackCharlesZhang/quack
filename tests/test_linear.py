# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch
import torch.nn.functional as F
import sys
import os

# Add the parent directory to the path to import quack modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quack.linear import linear_func, linear_act_func
from quack.gemm_interface import gemm_dact, gemm_dact_tuned, gemm_act_ref, gemm_dact_ref


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
    preact, postact = linear_act_func(
        x, w, activation, store_preact=store_preact, tuned=False
    )  # Disable tuning for faster test
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
    dx, postact = gemm_dact_tuned.fn(
        dout_input, weight.T, preact, activation=activation, config=None
    )
    dx_ref, postact_ref = gemm_dact_ref(
        dout_input.float(), weight.float().T, preact.float(), activation=activation
    )
    dx_pt, postact_pt = gemm_dact_ref(dout_input, weight.T, preact, activation=activation)
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-5
    assert (postact - postact_ref).abs().max() < 2 * (postact_pt - postact_ref).abs().max() + 1e-5
