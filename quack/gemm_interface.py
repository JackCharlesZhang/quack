# Copyright (c) 2025, Tri Dao
from typing import Optional, Tuple, Literal
from functools import partial
import sys
import os

import torch
import torch.nn.functional as F
from torch import Tensor

# Add the current directory to the path to import quack modules
sys.path.insert(0, os.path.dirname(__file__))

from quack.gemm_config import GemmConfig, get_all_configs

from quack.autotuner import autotune, AutotuneConfig
from quack.dense_gemm_sm90 import gemm_sm90
from quack.gemm_act_sm90 import gemm_act_sm90
from quack.gemm_dact_sm90 import gemm_dact_sm90


# Dictionary mapping activation names to PyTorch functions
act_to_pytorch_fn_map = {
    None: lambda x: x,
    "relu": F.relu,
    "relu_sq": lambda x: F.relu(x).square(),
    "gelu_tanh_approx": partial(F.gelu, approximate="tanh"),
}

# Dictionary mapping activation names to their gradient functions
# Each function takes (preact, dout) and returns (dx, postact)
dact_to_pytorch_fn_map = {
    None: lambda preact, dout: (dout, preact),
    "relu": lambda preact, dout: (
        torch.where(preact > 0, dout, torch.zeros_like(dout)),
        F.relu(preact),
    ),
    "relu_sq": lambda preact, dout: (
        torch.where(preact > 0, 2 * preact * dout, torch.zeros_like(dout)),
        F.relu(preact).square(),
    ),
    "gelu_tanh_approx": lambda preact, dout: (
        torch.autograd.grad(F.gelu(preact, approximate="tanh"), preact, dout, create_graph=False)[
            0
        ],
        F.gelu(preact, approximate="tanh"),
    ),
}

# Dictionary mapping gated activation names to their forward functions
# Each function takes (gate, up) and returns postact
gated_to_pytorch_fn_map = {
    "swiglu": lambda gate, up: F.silu(gate) * up,
    "swiglu_oai": lambda gate, up: gate * torch.sigmoid(1.702 * gate) * (up + 1),
    "reglu": lambda gate, up: F.relu(gate) * up,
    "geglu": lambda gate, up: F.gelu(gate, approximate="tanh") * up,
    "glu": lambda gate, up: torch.sigmoid(gate) * up,
}


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["out_dtype", "dynamic_scheduler"],
)
def gemm_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    C: Optional[Tensor] = None,  # (M, N)
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Optional[Tensor]):
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    if C is not None:
        C = C.unsqueeze(0)  # (1, M, N)
    out_dtype = A.dtype if out_dtype is None else out_dtype
    D = torch.empty((1, A.shape[1], B.shape[1]), dtype=out_dtype, device=A.device)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    )
    gemm_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        tile_count_semaphore,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
    )
    return D.squeeze(0)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "out_dtype", "postact_dtype", "store_preact"],
)
def gemm_act_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    C: Optional[Tensor] = None,  # (M, N)
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Optional[Tensor]):
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    if C is not None:
        C = C.unsqueeze(0)  # (1, M, N)
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    if store_preact:
        D = torch.empty((1, A.shape[1], B.shape[1]), dtype=out_dtype, device=A.device)
    else:
        D = None
    PostAct = torch.empty((1, A.shape[1], B.shape[1]), dtype=postact_dtype, device=A.device)
    gemm_act_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
    )
    return D.squeeze(0) if D is not None else None, PostAct.squeeze(0)


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "out_dtype", "postact_dtype", "dynamic_scheduler"],
)
def gemm_dact_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    PreAct: Tensor,  # (M, N)
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = True,
    config: Optional[GemmConfig] = None,
) -> (Tensor, Tensor):
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    PreAct = PreAct.unsqueeze(0)  # (1, M, N)
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    D = torch.empty((1, A.shape[1], B.shape[1]), dtype=out_dtype, device=A.device)
    PostAct = torch.empty((1, A.shape[1], B.shape[1]), dtype=postact_dtype, device=A.device)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    )
    gemm_dact_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        PreAct if not config.swap_ab else PreAct.mT,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
    )
    return D.squeeze(0), PostAct.squeeze(0)


@torch.library.custom_op("quack::gemm", mutates_args=(), device_types="cuda")
def gemm(
    A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None, dynamic_scheduler: bool = False
) -> Tensor:
    return gemm_tuned(A, B, None, out_dtype, dynamic_scheduler)


@torch.library.register_fake("quack::gemm")
def gemm_ref(
    A: Tensor, B: Tensor, out_dtype: Optional[torch.dtype] = None, dynamic_scheduler: bool = False
) -> Tensor:
    out_dtype = A.dtype if out_dtype is None else out_dtype
    return torch.mm(A, B).to(out_dtype)


@torch.library.custom_op("quack::gemm_add", mutates_args=(), device_types="cuda")
def gemm_add(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
) -> Tensor:
    return gemm_tuned(A, B, C, out_dtype, dynamic_scheduler)


@torch.library.register_fake("quack::gemm_add")
def gemm_add_ref(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
) -> Tensor:
    out_dtype = A.dtype if out_dtype is None else out_dtype
    return (C + torch.mm(A, B)).to(out_dtype)


@torch.library.custom_op("quack::gemm_add_t", mutates_args=(), device_types="cuda")
def gemm_t_add(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_tuned(A, B.T, C)


@torch.library.register_fake("quack::gemm_add_t")
def gemm_t_add_ref(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return gemm_add_ref(A, B.T, C)


# Specifying the schema manually here since torch.library._infer_schema doesn't work when return
# type is a tuple of Tensor
@torch.library.custom_op(
    "quack::gemm_act",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor? C=None, str? activation=None, ScalarType? out_dtype=None, ScalarType? postact_dtype=None, bool store_preact=True) -> (Tensor?, Tensor)",
)
def gemm_act(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
) -> Tuple[Tensor, Tensor]:
    return gemm_act_tuned(A, B, C, activation, out_dtype, postact_dtype, store_preact)


@torch.library.register_fake("quack::gemm_act")
def gemm_act_ref(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
) -> Tuple[Tensor, Tensor]:
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    out = torch.mm(A, B) if C is None else C + torch.mm(A, B)
    postact = act_to_pytorch_fn_map[activation](out).to(postact_dtype)
    return out.to(out_dtype) if store_preact else None, postact


# Specifying the schema manually here since torch.library._infer_schema doesn't work when return
# type is a tuple of Tensor
@torch.library.custom_op(
    "quack::gemm_dact",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor PreAct, str? activation=None, ScalarType? out_dtype=None, ScalarType? postact_dtype=None, bool dynamic_scheduler=True) -> (Tensor, Tensor)",
)
def gemm_dact(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = True,
) -> Tuple[Tensor, Tensor]:
    return gemm_dact_tuned(A, B, PreAct, activation, out_dtype, postact_dtype, dynamic_scheduler)


@torch.library.register_fake("quack::gemm_dact")
def gemm_dact_ref(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Reference implementation for GEMM with activation gradient."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    dout = torch.mm(A, B).to(out_dtype)
    dx, postact = dact_to_pytorch_fn_map[activation](PreAct, dout)
    return dx.to(out_dtype), postact.to(postact_dtype)


def gemm_gated_ref(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal["glu", "swiglu", "swiglu_oai", "reglu", "geglu"] = "swiglu",
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    """Reference implementation for GEMM with gated activation forward.

    Args:
        A: (M, K) - input tensor
        B: (K, 2*N) - weight tensor with gate and up projections
        C: (M, 2*N) - optional bias tensor
        activation: Type of gated activation
        out_dtype: Output dtype for preact
        postact_dtype: Output dtype for postact
        store_preact: Whether to return the pre-activation

    Returns:
        (preact, postact) where:
        - preact: (M, 2*N) pre-activation (if store_preact=True, else None)
        - postact: (M, N) post-activation output
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    preact = torch.mm(A, B) if C is None else C + torch.mm(A, B)
    # Split preact into gate and up projections
    gate = preact[..., ::2]  # (M, N)
    up = preact[..., 1::2]  # (M, N)
    postact = gated_to_pytorch_fn_map[activation](gate, up)
    return preact.to(out_dtype) if store_preact else None, postact.to(postact_dtype)


def gemm_dgated_ref(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal["glu", "swiglu", "swiglu_oai", "reglu", "geglu"],
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference implementation for GEMM with gated activation gradient.

    Args:
        A: (M, K) - dout input tensor
        B: (K, N) - weight tensor
        PreAct: (M, 2*N) - pre-activation tensor with gate and up projections interleaved
        activation: Type of gated activation
        out_dtype: Output dtype for dx
        postact_dtype: Output dtype for postact

    Returns:
        (dx, postact) where:
        - dx: (M, 2*N) gradient w.r.t. PreAct
        - postact: (M, N) post-activation output
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    dout = torch.mm(A, B).to(out_dtype)
    # Split PreAct into gate and up projections
    gate = PreAct[..., ::2]  # (M, N)
    up = PreAct[..., 1::2]  # (M, N)
    postact = gated_to_pytorch_fn_map[activation](gate, up)
    # Use autograd to compute gradients w.r.t. gate and up
    dgate, dup = torch.autograd.grad(postact, [gate, up], dout, create_graph=False)
    # Interleave gradients back
    dx = torch.stack([dgate, dup], dim=-1).reshape(PreAct.shape)
    return dx.to(out_dtype), postact.to(postact_dtype)


def gemm_dswiglu_ref(A: Tensor, B: Tensor, preact: Tensor) -> (Tensor, Tensor):
    # A: (M, K), B: (K, N), preact: (M, 2 * N)
    dout = torch.mm(A, B)
    p0, p1 = preact[..., ::2], preact[..., 1::2]
    sigmoid = torch.sigmoid(p0)
    silu = F.silu(p0)
    postact = silu * p1
    d0 = sigmoid * (1 + p0 * (1 - sigmoid)) * p1 * dout
    d1 = F.silu(p0) * dout
    out = torch.stack([d0, d1], dim=-1).reshape(d0.shape[:-1] + (2 * d0.shape[-1],))
    return out, postact
