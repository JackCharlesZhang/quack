# MLP + `torch.utils.checkpoint` (activation checkpointing)

## Summary

QuACK's `MLP` / `mlp_func` works with `torch.utils.checkpoint` out of the box.
Checkpointing eliminates saved activations during forward — only the input is retained.
The benefit scales with the number of layers in a multi-layer model.

## Background

Without checkpointing, the MLP forward saves two things for backward:
- **fc1** (`LinearActFunc`): saves `x` (input to fc1) — a view of the original input, so ~0 extra memory
- **fc2** (`DActLinearFunc`): saves `preact` (fc1's pre-activation output) — this is the big one

Note: fc2 already partially recomputes — it stores `preact` instead of `postact` and recomputes
`postact` from `preact` during backward via `gemm_dact`. So the only "extra" activation held
across forward→backward is `preact`.

With checkpointing, `preact` is not saved during forward. During backward, the entire forward
is replayed to recreate the autograd graph, then backward runs normally.

## Results (H100, bf16, 4×2048×4096 input, hidden=16384, 4 layers)

```
                                    NO checkpoint    WITH checkpoint
forward activation memory              1342 MB           268 MB      (saved 1074 MB)
peak memory                             3221 MB          3020 MB      (saved  201 MB)
```

- **Forward savings = N_layers × preact_size** (268 MB × 4 = 1074 MB)
- Peak savings are smaller because backward recomputation temporarily recreates activations

## Minimal repro

```python
import gc
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from quack.mlp import MLP

device = "cuda"
dtype = torch.bfloat16
B, S, D, hidden, N_LAYERS = 4, 2048, 4096, 16384, 4


class StackedMLP(nn.Module):
    def __init__(self, n_layers, use_checkpoint=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLP(D, hidden, activation="gelu_tanh_approx", device=device, dtype=dtype)
             for _ in range(n_layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint:
                x = cp.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


def run(use_checkpoint):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    model = StackedMLP(N_LAYERS, use_checkpoint=use_checkpoint)
    x = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    baseline = torch.cuda.memory_allocated()
    out = model(x)
    act_mem = (torch.cuda.memory_allocated() - baseline) / 1e6
    (out ** 2).sum().backward()  # NOTE: out.sum() won't work, see caveat below
    peak = torch.cuda.max_memory_allocated() / 1e6
    tag = "checkpoint" if use_checkpoint else "no checkpoint"
    print(f"[{tag:13s}] activations: {act_mem:7.0f} MB  peak: {peak:7.0f} MB")
    del model, x, out


run(use_checkpoint=False)
run(use_checkpoint=True)
```

## Usage

Wrap each MLP layer in your model's forward:

```python
# In a transformer block:
def forward(self, x):
    x = x + torch.utils.checkpoint.checkpoint(self.mlp, x, use_reentrant=False)
    return x
```

Or use the functional API:

```python
from quack.mlp import mlp_func

def forward(self, x):
    def _mlp(x):
        return mlp_func(x, self.fc1.weight, self.fc2.weight, activation="swiglu")
    x = x + torch.utils.checkpoint.checkpoint(_mlp, x, use_reentrant=False)
    return x
```

## `activation_memory_budget` (torch.compile SAC) — does NOT apply here

`torch._functorch.config.activation_memory_budget` is a separate mechanism:
- Requires `torch.compile` to trace the joint forward-backward graph
- Uses a min-cut partitioner to auto-decide which ops to save vs recompute (budget 0.0–1.0)
- Cannot see inside custom `autograd.Function` subclasses — QuACK's `LinearActFunc`,
  `DActLinearFunc`, etc. are opaque to the compiler

Since QuACK's MLP uses custom autograd Functions with hand-written `save_for_backward` /
`backward`, the `activation_memory_budget` knob has **no effect**. Use `torch.utils.checkpoint`
(manual wrapping) instead — it works because it operates at the Python level, replaying the
forward call entirely rather than trying to analyze the graph.

| Approach | Requires torch.compile | Works with custom autograd.Function | Granularity |
|---|---|---|---|
| `activation_memory_budget` | Yes | No (opaque) | Per-op (auto) |
| `torch.utils.checkpoint` | No | Yes | Per-layer (manual) |

## Caveat: `out.sum().backward()` produces zero-stride gradients

`out.sum()` produces an all-ones gradient via `expand()`, which has zero strides `(0, 0)`.
The GEMM kernels expect contiguous strides and will fail with:
```
ValueError: Mismatched mA.strides[0] ... expected to be 1
```

This only happens when the loss gradient is a broadcast tensor. Real losses (cross-entropy,
MSE, etc.) produce contiguous gradients. Workaround: use `(out ** 2).sum()` or any loss that
produces a non-trivial gradient.

A proper fix would be to add `dout = dout.contiguous()` at the top of `DActLinearFunc.backward`
(and similarly for `DGatedLinearFunc`).
