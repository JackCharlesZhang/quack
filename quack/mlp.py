# Copyright (c) 2025, Tri Dao
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange

from quack.linear import linear_act_func, act_linear_func
from quack.linear import linear_gated_func, gated_linear_func
from quack.gemm_act import gate_fn_map
from quack.gemm_interface import act_to_pytorch_fn_map, gated_to_pytorch_fn_map

Activation = Literal[
    "gelu_tanh_approx",
    "relu",
    "relu_sq",
    "swiglu",
    "swiglu_oai",
    "reglu",
    "geglu",
    "glu",
]


def mlp_func(x, weight1, weight2, activation: str, fuse_grad_accum=False, tuned=True):
    gated = activation in gate_fn_map
    fc1_fn = linear_gated_func if gated else linear_act_func
    fc2_fn = gated_linear_func if gated else act_linear_func
    preact, postact = fc1_fn(
        x,
        weight1,
        activation,
        store_preact=torch.is_grad_enabled(),
        fuse_grad_accum=fuse_grad_accum,
        tuned=tuned,
    )
    out = fc2_fn(
        preact,
        weight2,
        postact,
        activation=activation,
        fuse_grad_accum=fuse_grad_accum,
        tuned=tuned,
    )
    return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=False,
        bias2=False,
        activation: Activation = "gelu_tanh_approx",
        multiple_of=1,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
        tuned: bool = True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.activation = activation
        self.gated = activation in gate_fn_map
        if hidden_features is None:
            hidden_features = int(8 / 3 * in_features) if self.gated else 4 * in_features
        if multiple_of > 1:
            hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        fc1_out = 2 * hidden_features if self.gated else hidden_features
        self.fc1 = nn.Linear(in_features, fc1_out, bias=bias1, **factory_kwargs)
        if self.gated:
            self.fc1.weight._muon_reshape_functions = (
                lambda w: rearrange(w, "(d two) e -> two d e", two=2),
                lambda w: rearrange(w, "two d e -> (d two) e"),
            )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.fuse_grad_accum = fuse_grad_accum
        self.tuned = tuned

    def forward(self, input: Tensor) -> Tensor:
        if (
            self.fc1.bias is None
            and self.fc2.bias is None
            and input.is_cuda
            and input.stride(-1) == 1
            and self.fc1.in_features % 8 == 0
            and self.fc1.out_features % (16 if self.gated else 8) == 0
            and self.fc2.out_features % 8 == 0
        ):
            return mlp_func(
                input,
                self.fc1.weight,
                self.fc2.weight,
                activation=self.activation,
                fuse_grad_accum=self.fuse_grad_accum,
                tuned=self.tuned,
            )
        else:
            y = self.fc1(input)
            if self.gated:
                y = gated_to_pytorch_fn_map[self.activation](y[..., ::2], y[..., 1::2])
            else:
                y = act_to_pytorch_fn_map[self.activation](y)
            return self.fc2(y)
