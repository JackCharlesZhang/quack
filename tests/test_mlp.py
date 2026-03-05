# Copyright (C) 2025, Tri Dao.
import pytest
import torch
import torch.nn.functional as F

from quack.mlp import MLP
from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "relu", "swiglu", "reglu", "geglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_features", [512])
@pytest.mark.parametrize("in_features", [512])
def test_mlp(in_features, hidden_features, dtype, activation):
    device = "cuda"
    torch.random.manual_seed(0)
    batch = 256
    mlp = MLP(
        in_features, hidden_features, activation=activation, device=device, dtype=dtype, tuned=False
    )
    gated = activation in gated_to_pytorch_fn_map
    if gated:
        assert mlp.fc1.out_features == 2 * hidden_features
        assert mlp.fc2.in_features == hidden_features
    x = torch.randn(batch, in_features, device=device, dtype=dtype, requires_grad=True)
    out = mlp(x)
    assert out.shape == (batch, in_features)
    # Reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    y_ref = F.linear(x_ref, w1_ref)
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    assert (out.float() - out_ref).abs().max() < 1e-2
    # Backward
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout.float())
    assert x.grad is not None
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2
