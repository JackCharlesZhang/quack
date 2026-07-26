# Copyright (c) 2026, Tri Dao.
"""The single torch custom op for all epilogue-GEMM objects (HANDOFF Tier 4).

``quack::gemm_epi(str digest, Tensor[] ins, Tensor(a!)[] outs, str meta)``:
one mutating op + one no-op fake covers every epilogue — including
user-defined ``@gemm_epilogue`` fns, which otherwise have no torch.compile
story (precedent: torch.compile's own triton_kernel_wrapper_mutation).
``digest`` resolves the epilogue through an in-process registry, falling back
to import by the module locator carried in ``meta`` (so compiled artifacts
survive process boundaries when the epilogue is bound to an importable name —
the same constraint the async-compile pool imposes).

``ins`` is a packed list of the non-None input tensors, named positionally by
``meta['ins_names']``; ``outs`` carries D + declared outputs + reduce-partial
buffers (all graph-owned, since the op only mutates). Host constants (config,
flags, scalar operands) ride ``meta`` as a repr'd dict — Dynamo guards on the
string, which is exactly right since they select compiled behavior.

Reduce sinks under torch.compile: the wrapper pins the config (the partial
buffers must be graph-allocated at exact shapes BEFORE the op runs, so
runtime autotuning inside the op cannot pick a different tiling) and
finalizes the partials with traced torch ops.
"""

from __future__ import annotations

import ast
from typing import Optional

import torch

from quack.blockscaled.operand import BlockScaledFormat
from quack.gemm_config import GemmConfig, default_config
from quack.rounding import RoundingMode


def _sf_encode(SF: torch.Tensor) -> torch.Tensor:
    """e8m0 -> uint8 view across the mutable custom-op boundary (the Inductor
    decompose_auto_functionalized workaround; same seam as
    gemm_interface._sf_encode). Decoded format-driven in the op body."""
    return SF.view(torch.uint8) if SF.dtype == torch.float8_e8m0fnu else SF


def _sf_decode(SF, bs_format):
    if SF is not None and SF.dtype == torch.uint8:
        SF = SF.view(BlockScaledFormat.from_name(bs_format).scale_dtype)
    return SF


# digest -> EpiMod, populated by the compile-path wrapper (same process) or
# lazily by import through the meta locator.
_EPI_REGISTRY: dict = {}


def _resolve(digest: str, locator):
    mod = _EPI_REGISTRY.get(digest)
    if mod is None and locator:
        import importlib

        module = importlib.import_module(locator[0])
        mod = getattr(module, locator[1])
        if mod.semantic_digest != digest:
            raise RuntimeError(
                f"epilogue {locator[0]}.{locator[1]} changed since this graph was compiled"
            )
        _EPI_REGISTRY[digest] = mod
    if mod is None:
        raise RuntimeError(
            "epilogue digest not resolvable in this process; bind the @gemm_epilogue "
            "object to a module-global name in an importable module"
        )
    return mod


@torch.library.custom_op("quack::gemm_epi", mutates_args={"outs"}, device_types="cuda")
def _gemm_epi(digest: str, ins: list[torch.Tensor], outs: list[torch.Tensor], meta: str) -> None:
    m = ast.literal_eval(meta)
    mod = _resolve(digest, m["locator"])
    named = dict(zip(m["ins_names"], ins))
    operands = {k[4:]: v for k, v in named.items() if k.startswith("op__")}
    operands.update(m["scalar_ops"])
    i = 0
    out = {}
    if m["store_d"]:
        out["D"] = outs[0]
        i = 1
    for name in m["out_names"]:
        out[name] = outs[i]
        i += 1
    for name in m["sink_names"]:  # exact-shape partials: finalized by the wrapper
        operands[name] = outs[i]
        i += 1
    cfg = GemmConfig(**m["config"]) if m["config"] is not None else None
    mod(
        named["A"],
        named["B"],
        named.get("C"),
        out=out,
        store_d=m["store_d"],
        config=cfg,
        tuned=m["tuned"],
        cu_seqlens_m=named.get("cu_seqlens_m"),
        A_idx=named.get("A_idx"),
        SFA=_sf_decode(named.get("SFA"), m.get("bs_format_a")),
        SFB=_sf_decode(named.get("SFB"), m.get("bs_format_b")),
        bs_format_a=m.get("bs_format_a"),
        bs_format_b=m.get("bs_format_b"),
        rounding_mode=m["rounding_mode"],
        **operands,
    )


@_gemm_epi.register_fake
def _gemm_epi_fake(digest, ins, outs, meta) -> None:
    # Pure no-op: the op only mutates ``outs``; compilation is owned by
    # jit_cache + the async pool at real execution time.
    return


def _alloc_outs_from_meta(digest: str, ins: list, meta: str) -> list:
    """Allocate the graph-owned outs list ([D?] + declared outputs + reduce
    partials) from meta + ins alone. Shared by the functional op body and its
    fake: under FakeTensorMode the same torch.empty calls yield fakes, so the
    two sides cannot drift."""
    m = ast.literal_eval(meta)
    mod = _resolve(digest, m["locator"])
    named = dict(zip(m["ins_names"], ins))
    A, B, C = named["A"], named["B"], named.get("C")
    cu, A_idx = named.get("cu_seqlens_m"), named.get("A_idx")
    dt = getattr(torch, m["out_dtype"]) if m.get("out_dtype") else None
    out = mod._alloc_outputs(None, A, B, C, m["store_d"], dt, cu, A_idx)
    outs = ([out["D"]] if m["store_d"] else []) + [out[name] for name in m["out_names"]]
    if m["sink_names"]:
        cfg, lead, n = m["config"], mod._lead_shape(A, cu, A_idx), B.shape[-1]
        for name in m["sink_names"]:
            op = mod.sinks[name]
            if op.dim == 0:
                shape = (*lead, -(-n // cfg["tile_n"]))
            else:
                shape = (*lead[:-1], -(-lead[-1] // cfg["tile_m"]), n)
            outs.append(torch.empty(shape, dtype=torch.float32, device=A.device))
    return outs


@torch.library.custom_op("quack::gemm_epi_f", mutates_args=(), device_types="cuda")
def _gemm_epi_f(digest: str, ins: list[torch.Tensor], meta: str) -> list[torch.Tensor]:
    """Functional form of ``quack::gemm_epi``: outputs allocated inside, real
    fake — one graph-insertable node per epilogue-GEMM call. Graph-owned
    buffers only; caller-provided out=/partial buffers take the mutating op."""
    outs = _alloc_outs_from_meta(digest, ins, meta)
    torch.ops.quack.gemm_epi(digest, ins, outs, meta)
    return outs


@_gemm_epi_f.register_fake
def _gemm_epi_f_fake(digest, ins, meta):
    return _alloc_outs_from_meta(digest, ins, meta)


def compile_call(
    mod,
    A,
    B,
    C,
    *,
    out,
    out_dtype,
    store_d,
    config,
    tuned,
    cu_seqlens_m,
    A_idx,
    SFA,
    SFB,
    bs_format_a,
    bs_format_b,
    rounding_mode,
    operands,
):
    """torch.compile-path body of ``EpiMod.__call__``: record one functional
    ``quack::gemm_epi_f`` call (allocation inside the op, so the graph gets a
    single node) and finalize reduces with traced ops. Caller-provided out=/
    partial buffers cannot be graph-owned, so that case keeps the mutating
    ``quack::gemm_epi`` form. Returns the same dict as eager."""
    cfg: Optional[GemmConfig] = config
    if mod.sinks and cfg is None:
        # Partials are graph-allocated before the op runs, so the tiling must
        # be fixed here — no runtime autotune under compile with sinks.
        cfg = default_config(A.device)
        tuned = False
    caller_owned = bool(out) or any(operands.get(name) is not None for name in mod.sinks)
    sink_names = tuple(name for name in mod.sinks if operands.get(name) is None)

    ins_names, ins = [], []
    for name, t in (
        ("A", A),
        ("B", B),
        ("C", C),
        ("cu_seqlens_m", cu_seqlens_m),
        ("A_idx", A_idx),
        ("SFA", _sf_encode(SFA) if SFA is not None else None),
        ("SFB", _sf_encode(SFB) if SFB is not None else None),
        *((f"op__{k}", v) for k, v in operands.items() if isinstance(v, torch.Tensor)),
    ):
        if t is not None:
            ins_names.append(name)
            ins.append(t)
    scalar_ops = {k: v for k, v in operands.items() if not isinstance(v, torch.Tensor)}
    meta = repr(
        dict(
            ins_names=tuple(ins_names),
            out_names=tuple(mod.outputs),
            sink_names=sink_names,
            store_d=bool(store_d),
            tuned=bool(tuned),
            config=None if cfg is None else cfg.__dict__,
            rounding_mode=int(rounding_mode),
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            scalar_ops=scalar_ops,
            out_dtype=None if out_dtype is None else str(out_dtype).split(".")[-1],
            locator=mod._module_locator(),
        )
    )
    _EPI_REGISTRY[mod.semantic_digest] = mod
    if caller_owned:
        out = mod._alloc_outputs(out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx)
        lead = mod._lead_shape(A, cu_seqlens_m, A_idx)
        n = B.shape[-1]
        partials = {}
        for name in sink_names:
            op = mod.sinks[name]
            if op.dim == 0:
                shape = (*lead, -(-n // cfg.tile_n))
            else:
                shape = (*lead[:-1], -(-lead[-1] // cfg.tile_m), n)
            partials[name] = torch.empty(shape, dtype=torch.float32, device=A.device)
        outs = []
        if store_d:
            outs.append(out["D"])
        outs.extend(out[name] for name in mod.outputs)
        outs.extend(partials.values())
        torch.ops.quack.gemm_epi(mod.semantic_digest, ins, outs, meta)
    else:
        outs = torch.ops.quack.gemm_epi_f(mod.semantic_digest, ins, meta)
        i = 1 if store_d else 0
        out = {"D": outs[0]} if store_d else {}
        for j, name in enumerate(mod.outputs):
            out[name] = outs[i + j]
        i += len(mod.outputs)
        partials = {name: outs[i + j] for j, name in enumerate(sink_names)}
    result = dict(out) if store_d else {k: v for k, v in out.items() if k != "D"}
    for name, buf in partials.items():
        finalize = getattr(mod.sinks[name], "host_finalize", None)
        result[name] = finalize(buf) if finalize is not None else buf
    return result


_DEFAULT_RN = RoundingMode.RN  # re-export convenience for the __call__ branch
