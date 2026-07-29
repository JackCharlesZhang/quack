# Copyright (c) 2026, Tri Dao.
"""Host-side plumbing that makes A-operand transforms first-class in the
generic GEMM host layer (:mod:`quack.gemm_host`), mirroring what EpiOps'
host hooks do for epilogues:

* :class:`TransformARef` — a picklable reference that crosses the
  jit-cache / async-compile boundary (registered W4 format names resolve
  by name; fn-authored / instance mods resolve through a process-local
  registry keyed by their semantic digest, shipped to workers as a
  cloudpickle payload exactly like ``epi_mod_local`` in gemm_host).
* the W4 blob/strip geometry — ONE implementation (over meta tensors)
  serves both the runtime torch views and the trace-time fake tensors, so
  the compiled layout and the launched layout cannot drift.

A transform handle is anything ``transform_a=`` accepts today: a registered
format name, a DecodeFormat instance, or an ATransformMod / PackedFormatMod.
Value transforms (packed=None) have no host geometry — they only contribute
the ctor factory and their semantic digest to the compile key.
"""

from typing import NamedTuple, Optional

import torch

import cutlass
import cutlass.cute as cute

from quack.blockscaled.decode_formats import DecodeFormat, W4_FORMATS
from quack.operand_transform.transform_a import TransformAOperand

__all__ = [
    "TransformARef",
    "transform_a_ref",
    "resolve_transform_a",
    "transform_decode_format",
    "w4_operand_views",
    "w4_fake_operands",
    "transform_a_operand",
    "transform_a_fake_operand",
]


class TransformARef(NamedTuple):
    """Picklable recipe for resolving a transform mod in async workers.

    ``w4_name`` re-mints from the W4_FORMATS registry by name; ``mod_local``
    resolves through the process-local registry below (populated by
    ``transform_a_ref`` before the compile that needs it, and by
    ``install_transform_mod_payload`` in async workers)."""

    kind: str  # "w4_name" | "mod_local"
    name: str = ""
    semantic_digest: str = ""

    def __quack_pool_payload__(self):
        if self.kind != "mod_local":
            return None
        import cloudpickle

        from quack.cache.async_compile import PoolPayload

        payload = cloudpickle.dumps(_LOCAL_TRANSFORM_MODS[self.semantic_digest])
        return PoolPayload(
            "quack.operand_transform.host",
            "install_transform_mod_payload",
            self.semantic_digest,
            payload,
        )


# semantic_digest -> mod. Entries are small (format bundles / fn mods), so
# unlike gemm_host's epi registry they are not consumed on resolution — a
# plan may compile several configs against the same mod.
_LOCAL_TRANSFORM_MODS: dict[str, object] = {}


def install_transform_mod_payload(expected_digest: str, data: bytes) -> None:
    import cloudpickle

    mod = cloudpickle.loads(data)
    if mod.semantic_digest != expected_digest:
        raise ValueError(
            f"transform payload digest mismatch: expected {expected_digest}, "
            f"got {mod.semantic_digest}"
        )
    _LOCAL_TRANSFORM_MODS[expected_digest] = mod


def transform_a_ref(handle):
    """Normalize a transform handle to ``(ref, mod)``: the picklable ref for
    the compile key and the resolved mod for immediate use."""
    from quack.operand_transform.frontend import (
        ATransformMod,
        DropoutAMod,
        PackedFormatMod,
        w4_transform,
    )

    if isinstance(handle, str):
        if handle not in W4_FORMATS:
            raise KeyError(f"unknown W4 format {handle!r}")
        return TransformARef("w4_name", name=handle), w4_transform(handle)
    if isinstance(handle, DecodeFormat):
        handle = PackedFormatMod(handle)
    if not isinstance(handle, (ATransformMod, PackedFormatMod, DropoutAMod)):
        raise TypeError(f"not a transform handle: {handle!r}")
    _LOCAL_TRANSFORM_MODS[handle.semantic_digest] = handle
    return TransformARef("mod_local", semantic_digest=handle.semantic_digest), handle


def resolve_transform_a(ref: TransformARef):
    if ref.kind == "w4_name":
        from quack.operand_transform.frontend import w4_transform

        return w4_transform(ref.name)
    if ref.kind != "mod_local":
        raise ValueError(f"unknown transform reference kind {ref.kind!r}")
    mod = _LOCAL_TRANSFORM_MODS.get(ref.semantic_digest)
    if mod is None:
        raise RuntimeError(
            "process-local transform reference is not registered here (created in "
            "another process and its payload was not installed)"
        )
    return mod


def transform_decode_format(mod) -> Optional[DecodeFormat]:
    """The DecodeFormat behind a layout-owning transform mod, or None for
    value transforms (which keep A a plain (M, K) operand)."""
    packed = getattr(mod, "packed", None)
    if packed is None:
        return None
    if isinstance(packed, DecodeFormat):
        return packed  # PackedFormatMod
    return mod.as_decode_format()  # ATransformMod with PackedInput


def transform_handle_fmt(handle) -> Optional[DecodeFormat]:
    """The DecodeFormat behind ANY transform handle (name / format instance /
    mod), or None for value transforms. Cheap — no digest hashing — so it is
    safe on warm plan-cache paths."""
    if isinstance(handle, str):
        if handle not in W4_FORMATS:
            raise KeyError(f"unknown W4 format {handle!r}")
        return W4_FORMATS[handle]
    if isinstance(handle, DecodeFormat):
        return handle
    return transform_decode_format(handle)


def transform_handle_key(handle):
    """Cheap plan-cache identity for a transform handle: registry name, a
    mod's precomputed semantic digest, or the instance id (module-level
    format instances key by identity, like pinned GemmConfigs)."""
    if isinstance(handle, str):
        return ("w4_name", handle)
    digest = getattr(handle, "semantic_digest", None)
    if digest is not None:
        return ("mod", digest)
    return ("fmt_id", id(handle))


# ---- fn-operand host geometry (one entry per kind in ----------------------
# ---- transform_a.A_TRANSFORM_ARG_KINDS) ------------------------------------


def _strip_dims(gran_m, gran_k, tile_m, tile_k):
    """Host mirror of transform_a._strip_geometry: (gran_m, g_m, gran_k,
    g_k, k_inner). The box's stride-1 axis is the finer one (TMA needs 16 B
    alignment on every non-inner stride)."""
    gran_m = tile_m if gran_m is None else gran_m
    gran_k = tile_k if gran_k is None else gran_k
    assert tile_m % gran_m == 0, f"gran_m {gran_m} must divide tile_M ({tile_m})"
    assert tile_k % gran_k == 0, f"gran_k {gran_k} must divide tile_K ({tile_k})"
    return gran_m, tile_m // gran_m, gran_k, tile_k // gran_k, gran_k < gran_m


def _strip_view(gran_m, gran_k):
    """View builder for a (gran_m, gran_k) strip: the user tensor is
    (outer-groups, inner-groups) row-major with the FINER axis contiguous —
    (rk * g_k, M / gran_m) for m-fine strips (colvec family), (M / gran_m,
    rk * g_k) for k-fine strips (kvec family); rk = ceil(K / tile_k), so a
    ragged K tail is PADDED to whole k-tiles at group granularity. Returns
    the (inner, outer, G_m, rk, 1) element-typed VIEW for the aux TMA slot
    (one box per (m-tile, k-tile) — see transform_a._StripAux)."""

    def view(A, value, tile_m, tile_k):
        granm, g_m, grank, g_k, k_inner = _strip_dims(gran_m, gran_k, tile_m, tile_k)
        m, k = A.shape
        assert value.dtype == A.dtype and value.element_size() == 2
        assert value.is_contiguous(), "strip operands need a contiguous (outer, inner) tensor"
        assert m % tile_m == 0, f"M ({m}) must be divisible by tile_M ({tile_m})"
        rk = -(-k // tile_k)
        mg, kg = m // granm, rk * g_k
        shape = (mg, kg) if k_inner else (kg, mg)
        assert value.shape == torch.Size(shape), (
            f"strip shape {tuple(value.shape)} must be {shape} = "
            f"(M / {granm}, ceil(K / tile_K) * {g_k})"
            + ("" if k_inner else " transposed")
            + " (K padded to whole k-tiles)"
        )
        if k_inner:
            # (g_k, g_m, Gm, rk, 1); strides (1, KG, g_m*KG, g_k, 1)
            return value.view(m // tile_m, g_m, rk, g_k).permute(3, 1, 0, 2).unsqueeze(-1)
        # (g_m, g_k, Gm, rk, 1); strides (1, MG, g_m, g_k*MG, 1)
        return value.view(rk, g_k, m // tile_m, g_m).permute(3, 1, 2, 0).unsqueeze(-1)

    return view


def _strip_fake(gran_m, gran_k):
    """Fake builder matching :func:`_strip_view`'s strides exactly:
    contiguous inner run, one static stride (the box-outer axis's per-m-tile
    / per-k-tile step within the user tensor), symbolic G_m / rk extents
    and symbolic remaining strides (8-element-divisible: the 16 B TMA rule,
    guaranteed by M % tile_M == 0 and g_k >= 8 on the k-fine path)."""

    def fake(a_dtype, tile_m, tile_k):
        _granm, g_m, _grank, g_k, k_inner = _strip_dims(gran_m, gran_k, tile_m, tile_k)
        gm, rk = cute.sym_int(), cute.sym_int()
        sym8 = lambda: cute.sym_int64(divisibility=8)
        if k_inner:
            shape, stride = (g_k, g_m, gm, rk, 1), (1, sym8(), sym8(), g_k, 1)
        else:
            shape, stride = (g_m, g_k, gm, rk, 1), (1, sym8(), g_m, sym8(), 1)
        return cute.runtime.make_fake_tensor(a_dtype, shape, stride=stride, assumed_align=16)

    return fake


def _strip_host(gran_m, gran_k):
    return (_strip_view(gran_m, gran_k), _strip_fake(gran_m, gran_k))


def _seed_view(A, value, tile_m, tile_k):
    """The dropout [seed, offset] operand: a (2,) int64 CUDA tensor crossing
    RAW in the bundle's sf slot (aux_raw — no TMA box, no smem)."""
    assert value.dtype == torch.int64 and tuple(value.shape) == (2,), (
        "seed operand must be a (2,) int64 [seed, offset] tensor"
    )
    assert value.is_cuda and value.is_contiguous() and value.data_ptr() % 16 == 0
    return value


def _seed_fake(a_dtype, tile_m, tile_k):
    return cute.runtime.make_fake_tensor(cutlass.Int64, (2,), stride=(1,), assumed_align=16)


_ARG_KIND_HOST = {
    "colvec_ktile": _strip_host(1, None),
    "colvec_k64": _strip_host(1, 64),
    "colvec_k32": _strip_host(1, 32),
    "colvec_k16": _strip_host(1, 16),
    "kvec_m64": _strip_host(64, 1),
    "seed_i64x2": (_seed_view, _seed_fake),
}


def transform_a_operand(mod, A, values: dict, tile_m: int, tile_k: int = 64) -> TransformAOperand:
    """The kernel's A operand for a value transform with runtime operands: a
    TransformAOperand bundle of the plain (M, K) A and the operand views the
    mod's ``args`` declare (kind-dispatched; one aux-delivered operand for
    now). ``values`` maps fn param name -> tensor; ``tile_m``/``tile_k`` must
    match the launched config."""
    args = getattr(mod, "args", ())
    assert args, f"{getattr(mod, 'name', mod)!r} declares no runtime operands (args=)"
    assert set(values) == {name for name, _ in args}, (
        f"operand values {set(values)} must match the declared args {[n for n, _ in args]}"
    )
    assert A.ndim == 2, "runtime operands support the plain dense path"
    views = [_ARG_KIND_HOST[kind][0](A, values[name], tile_m, tile_k) for name, kind in args]
    assert len(views) == 1, "one aux-delivered operand per transform (single aux slot)"
    return TransformAOperand(A, views[0])


def transform_a_fake_operand(mod, mA_fake, a_dtype, tile_m: int, tile_k: int) -> TransformAOperand:
    """Trace-time counterpart of :func:`transform_a_operand`."""
    args = getattr(mod, "args", ())
    fakes = [_ARG_KIND_HOST[kind][1](a_dtype, tile_m, tile_k) for _name, kind in args]
    assert len(fakes) == 1, "one aux-delivered operand per transform (single aux slot)"
    return TransformAOperand(mA_fake, fakes[0])


# ---- W4 blob / strip geometry ----------------------------------------------


def _w4_views(fmt: DecodeFormat, blob_u8, sf_u8, tile_m: int):
    """The kernel-facing views: blob (g, kt, 128, wpt*... bytes) ->
    (256, wpt, tm64, Gt, Kt, 1) with a 256 B contiguous TMA inner run, and
    the SF strip -> (sfb, tm64, Gt, Kt, 1). Works on real AND meta tensors —
    the fake path below reuses it so trace and launch layouts match by
    construction."""
    tm64 = tile_m // 64
    g, kt = blob_u8.shape[:2]
    gt = g // tm64
    wpt = (16 if fmt.w8 else 8) * (fmt.tile_k // 64)  # 256 B runs per (m64, k-tile)
    mA_t = blob_u8.view(gt, tm64, kt, wpt, 256).permute(4, 3, 1, 0, 2).unsqueeze(-1)
    mSFA_t = None
    if fmt.sf_words > 0:
        sfb = fmt.sf_bytes
        mSFA_t = sf_u8.reshape(g, kt, sfb).view(gt, tm64, kt, sfb).permute(3, 1, 0, 2).unsqueeze(-1)
    return mA_t, mSFA_t


def w4_operand_views(fmt: DecodeFormat, blob, sf, tile_m: int) -> TransformAOperand:
    """The kernel's A operand: a TransformAOperand bundle of torch views
    (blob + optional SF strip). It crosses the boundary in the mA slot as one
    argument — the host layer never unpacks it."""
    assert (sf is not None) == (fmt.sf_words > 0), (
        f"format {fmt.name!r} takes {'a repacked SF blob' if fmt.sf_words else 'sf=None'}"
    )
    blob_u8 = blob.view(torch.uint8) if blob.dtype != torch.uint8 else blob
    sf_u8 = sf.view(torch.uint8) if sf is not None else None
    return TransformAOperand(*_w4_views(fmt, blob_u8, sf_u8, tile_m))


def w4_fake_operands(fmt: DecodeFormat, n_full: int, k: int, tile_m: int) -> TransformAOperand:
    """Trace-time fake bundle with the exact (static) shapes/strides the
    runtime views produce."""
    tk = fmt.tile_k
    assert n_full % 64 == 0 and k % tk == 0
    g, kt = n_full // 64, k // tk
    wpt = (16 if fmt.w8 else 8) * (tk // 64)
    blob_meta = torch.empty(g * kt * wpt * 256, dtype=torch.uint8, device="meta")
    sf_meta = None
    if fmt.sf_words > 0:
        sf_meta = torch.empty(g * kt * fmt.sf_bytes, dtype=torch.uint8, device="meta")
    mA_m, mSFA_m = _w4_views(
        fmt,
        blob_meta.view(g, kt, wpt * 256),
        sf_meta.view(g, kt, fmt.sf_bytes) if sf_meta is not None else None,
        tile_m,
    )
    fake = lambda t: cute.runtime.make_fake_tensor(
        cutlass.Uint8, tuple(t.shape), stride=tuple(t.stride()), assumed_align=16
    )
    return TransformAOperand(fake(mA_m), fake(mSFA_m) if mSFA_m is not None else None)
