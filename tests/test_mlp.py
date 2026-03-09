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


@pytest.mark.parametrize("use_compile", [False, True])
@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_zero_stride_grad(dtype, activation, use_compile):
    """out.sum().backward() produces expanded gradient with zero strides."""
    device = "cuda"
    torch.random.manual_seed(0)
    mlp = MLP(512, 512, activation=activation, device=device, dtype=dtype, tuned=False)
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    if use_compile:
        mlp = torch.compile(mlp)
    x = torch.randn(256, 512, device=device, dtype=dtype, requires_grad=True)
    out = mlp(x)
    out.sum().backward()
    assert x.grad is not None
    # Reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    y_ref = F.linear(x_ref, w1_ref)
    gated = activation in gated_to_pytorch_fn_map
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    out_ref.sum().backward()
    assert (x.grad.float() - x_ref.grad).abs().max() < 1e-2


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mlp_recompute(dtype, activation):
    """recompute=True should match normal mode and float32 reference."""
    device = "cuda"
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    gated = activation in gated_to_pytorch_fn_map
    mlp = MLP(dim, hidden, activation=activation, device=device, dtype=dtype, tuned=False)
    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch, dim, device=device, dtype=dtype)
    # Standard forward/backward
    out_std = mlp(x)
    out_std.backward(dout)
    dx_std = x.grad.clone()
    dw1_std = mlp.fc1.weight.grad.clone()
    dw2_std = mlp.fc2.weight.grad.clone()
    # Recompute forward/backward (toggle flag on same model)
    x.grad = None
    mlp.zero_grad()
    mlp.recompute = True
    out_rec = mlp(x)
    out_rec.backward(dout)
    dx_rec = x.grad.clone()
    dw1_rec = mlp.fc1.weight.grad.clone()
    dw2_rec = mlp.fc2.weight.grad.clone()
    mlp.recompute = False
    # vs normal mode
    assert (out_std - out_rec).abs().max() < 1e-6, "Output mismatch"
    assert (dx_std - dx_rec).abs().max() < 1e-2, "dx mismatch"
    assert (dw1_std - dw1_rec).abs().max() < 1e-2, "dW1 mismatch"
    assert (dw2_std - dw2_rec).abs().max() < 1e-2, "dW2 mismatch"
    # vs float32 reference
    x_ref = x.detach().clone().float().requires_grad_(True)
    w1_ref = mlp.fc1.weight.detach().clone().float()
    w2_ref = mlp.fc2.weight.detach().clone().float()
    y_ref = F.linear(x_ref, w1_ref)
    if gated:
        y_ref = gated_to_pytorch_fn_map[activation](y_ref[..., ::2], y_ref[..., 1::2])
    else:
        y_ref = act_to_pytorch_fn_map[activation](y_ref)
    out_ref = F.linear(y_ref, w2_ref)
    out_ref.backward(dout.float())
    assert (out_rec.float() - out_ref).abs().max() < 1e-2
    assert (dx_rec.float() - x_ref.grad).abs().max() < 1e-2


@pytest.mark.parametrize("activation", ["gelu_tanh_approx", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "freeze", ["x", "weight1", "weight2", "x_weight1", "x_weight2", "weight1_weight2"]
)
def test_mlp_recompute_partial_grad(dtype, activation, freeze):
    """Recompute mode with some inputs frozen (requires_grad=False).

    Tests that needs_input_grad indexing is correct: x=[0], weight1=[1], weight2=[2].
    """
    device = "cuda"
    torch.random.manual_seed(0)
    batch, dim, hidden = 256, 512, 512
    gated = activation in gated_to_pytorch_fn_map
    mlp = MLP(
        dim,
        hidden,
        activation=activation,
        device=device,
        dtype=dtype,
        tuned=False,
        recompute=True,
    )
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    dout = torch.randn(batch, dim, device=device, dtype=dtype)
    frozen = set(freeze.split("_")) if "_" in freeze else {freeze}
    freeze_x = "x" in frozen
    freeze_w1 = "weight1" in frozen
    freeze_w2 = "weight2" in frozen
    x.requires_grad_(not freeze_x)
    mlp.fc1.weight.requires_grad_(not freeze_w1)
    mlp.fc2.weight.requires_grad_(not freeze_w2)
    # Normal (non-recompute) forward/backward as reference
    mlp.recompute = False
    out_ref = mlp(x)
    out_ref.backward(dout)
    dx_ref = x.grad.clone() if x.grad is not None else None
    dw1_ref = mlp.fc1.weight.grad.clone() if mlp.fc1.weight.grad is not None else None
    dw2_ref = mlp.fc2.weight.grad.clone() if mlp.fc2.weight.grad is not None else None
    # Recompute forward/backward
    x.grad = None
    mlp.zero_grad()
    mlp.recompute = True
    out = mlp(x)
    out.backward(dout)
    dx = x.grad.clone() if x.grad is not None else None
    dw1 = mlp.fc1.weight.grad.clone() if mlp.fc1.weight.grad is not None else None
    dw2 = mlp.fc2.weight.grad.clone() if mlp.fc2.weight.grad is not None else None
    # Forward should match exactly
    assert (out - out_ref).abs().max() < 1e-6, "Output mismatch"
    # Gradients: recompute should match normal mode
    if freeze_x:
        assert dx is None and dx_ref is None
    else:
        assert (dx - dx_ref).abs().max() < 1e-2, "dx mismatch"
    if freeze_w1:
        assert dw1 is None and dw1_ref is None
    else:
        assert (dw1 - dw1_ref).abs().max() < 1e-2, "dW1 mismatch"
    if freeze_w2:
        assert dw2 is None and dw2_ref is None
    else:
        assert (dw2 - dw2_ref).abs().max() < 1e-2, "dW2 mismatch"
