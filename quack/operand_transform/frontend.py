# Copyright (c) 2026, Tri Dao.
"""``@a_transform``: author an A-operand transform as a plain Python function,
mirroring ``@gemm_epilogue`` (the fn is the COMPOSITION site; the kernel-side
plumbing — canonical s2r load or blob TMA, the interleaved produce/WGMMA
schedule, fences and commit groups — is written once and never by the fn).

Two families, one decorator:

* VALUE transforms (default): unpacked 16-bit A, canonically ldmatrix-loaded;
  the fn is called per lane per ``vec_size`` fragment elements as a TensorSSA
  vector in the MMA dtype and returns the transformed vector::

      @a_transform(vec_size=2)
      def halve_a(x):
          return x * 0.5

  The vector is FRAGMENT-SLOT-ORDERED, not k-contiguous (a lane's 8 elements
  per k16 block are 2 adjacent k x 2 rows x 2 k-halves); chunks are
  pair-aligned so packed 16-bit math vectorizes. ``vec_size`` in {2, 4, 8} and
  is capped at one k16 block: the schedule (produce(b+1) overlapping WGMMA(b))
  belongs to the framework, never the fn. Compile-time constants may be closed
  over — they are part of the semantic key. ``consts=callable`` is called once
  per kernel (hoisted — LUTs, packed constants); its result is the fn's LAST
  parameter.

  Runtime operands are declared per fn parameter with ``args={param: kind}``,
  a kind taxonomy over the transform's (M, K) index space — the mainloop
  mirror of EpiOps' operand kinds over (M, N). Each kind owns its indexing,
  delivery and register staging; the fn just receives a same-length TensorSSA
  vector of values aligned with x. Kinds (transform_a.A_TRANSFORM_ARG_KINDS):

  * the strip family — one MMA-dtype value per (m-group, k-group) at 2-D
    granularity, delivered via the aux A-side TMA slot (per-stage smem under
    the AB mbarrier): ``"colvec_ktile"`` (per (row, k-tile)),
    ``"colvec_k64"`` / ``"colvec_k32"`` / ``"colvec_k16"`` (per (row, 64 /
    32 / 16 elements) — dense blockscaled-SF granularities), ``"kvec_m64"``
    (per (m64 row block, k-element) — the LCE dw strip; resolve
    coarser-than-64 M granularity host-side, e.g. a vocab-tile row per m64
    block). E.g. the linear-CE dx pow2 rescale::

        @a_transform(vec_size=8, args={"u": "colvec_ktile"})
        def dx_scale(x, u):
            return x * u

    The host passes A as the bundle
    ``host.transform_a_operand(mod, A, {"u": strip}, tile_m, tile_k)`` with
    ``strip`` a contiguous (ceil(K / tile_K) * g, M) tensor in A's dtype
    (one row per k-group; ragged K padded to whole k-tiles).

  There is deliberately NO k-invariant ``colvec`` kind: a per-row scale
  commutes through the GEMM — ``(u ⊙ A) @ B = u ⊙ (A @ B)`` — so it belongs
  in the EPILOGUE as an fp32 colvec (exact, measured free), and per-row
  additive terms commute via the rank-1 ``z · colsum(B)`` correction; only a
  fn NONLINEAR in x would need per-row mainloop values.

* DROPOUT (``dropout_a(p)``): not a fn — a dedicated mask-only transform
  (philox keep-mask ANDed onto the fragment at ~3 SASS per 2 elements; see
  :class:`~quack.operand_transform.transform_a.TransformADropout`). The
  (2,) int64 [seed, offset] tensor rides the same bundle
  (``host.transform_a_operand(dropout_a(p), A, {"seed": t}, tile_m)``, or
  ``TransformAOperand(A, t)`` at the direct-compile layer). The mask of
  (m, k) is a pure function of (m, k, seed, offset) — any kernel
  regenerates it — and is split-k invariant. Mask-only: fold 1/(1-p) into
  the epilogue.

* PACKED decodes (``packed=PackedInput(...)``): the fn IS the decode — the
  :meth:`~quack.blockscaled.decode_formats.DecodeFormat.decode_k16` body,
  ``fn(xw, sfw, b, consts) -> 4 packed regs`` — and the ``PackedInput``
  carries the geometry (w8 / tile_k) and the host bundle (prepare /
  quantize_reference / dequant_reference) that must stay consistent with it.
  The mod mints a ``DecodeFormat``, so it slots into everything the
  class-based formats can (TransformAW4, the gemm_w4 wrapper, the roundtrip
  test fixture). ``sfw`` is always None for now (scale-factor strips need the
  aux A-side operand, not ported yet — sf_words must be 0).

A mod is a factory ``gemm -> TransformA`` — pass it straight to
``GemmSm90(transform_a=mod)``. ``__quack_semantic_key__`` fail-closed
fingerprints the fn (source + every capture, via the gemm_epilogue keyer), so
mods compose with the jit-cache machinery.
"""

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable, Optional


import cutlass.cute as cute

from quack.blockscaled.decode_formats import DecodeFormat, decode_format
from quack.gemm_epilogue import _function_semantic_key, _semantic_value_key
from quack.operand_transform.transform_a import TransformAValue, TransformAW4

__all__ = ["a_transform", "ATransformMod", "PackedInput", "dropout_a", "w4_transform"]


@dataclass(frozen=True)
class PackedInput:
    """Storage geometry + host bundle for a packed-storage fn transform.
    Field meanings match :class:`~quack.blockscaled.decode_formats.DecodeFormat`
    (w8: 32 B/thread raw words; tile_k: the k-tile the repack is built
    around). The host callables must stay consistent with the fn — the
    roundtrip test fixture is what pins that."""

    name: str
    w8: bool = False
    tile_k: int = 64
    make_consts: Optional[Callable] = None
    prepare: Optional[Callable] = None
    quantize_reference: Optional[Callable] = None
    dequant_reference: Optional[Callable] = None


class ATransformMod:
    """A fn-authored A-operand transform; callable as the ``transform_a=``
    factory. See module docstring for the fn contracts."""

    def __init__(self, fn, vec_size, packed, consts=None, regs=None, args=None):
        self.fn = fn
        self.name = getattr(fn, "__name__", "a_transform")
        self.packed = packed
        if packed is not None:
            assert vec_size in (None, 8), "packed fn transforms decode whole k16 blocks"
            assert consts is None, "packed fns take consts via PackedInput.make_consts"
            assert not args, "runtime operands are value-fn only (packed decodes own A)"
            vec_size = 8
        else:
            vec_size = 2 if vec_size is None else vec_size
            assert vec_size in (2, 4, 8), "vec_size must be 2, 4 or 8 (one k16 block max)"
        self.vec_size = vec_size
        self.consts = consts
        self.regs = regs
        self.args = _normalize_args(fn, args)
        self._fmt = None
        self.semantic_digest = _digest(self.__quack_semantic_key__())

    def __call__(self, gemm):
        if self.packed is not None:
            return TransformAW4(gemm, self.as_decode_format())
        return TransformAValue(gemm, self)

    def as_decode_format(self) -> DecodeFormat:
        """Mint the DecodeFormat backing a packed fn transform (cached)."""
        assert self.packed is not None, "value transforms do not define a decode format"
        if self._fmt is None:
            mod, spec = self, self.packed

            class _FnFormat(DecodeFormat):
                name = spec.name
                w8 = spec.w8
                tile_k = spec.tile_k

                def make_consts(self):
                    return spec.make_consts() if spec.make_consts is not None else None

                @cute.jit
                def decode_k16(self, xw, sfw, b, consts):
                    return mod.fn(xw, sfw, b, consts)

                def quantize_reference(self, w):
                    return spec.quantize_reference(w)

                def dequant_reference(self, q, sf):
                    return spec.dequant_reference(q, sf)

                def prepare(self, q, sf):
                    return spec.prepare(q, sf)

            _FnFormat.__qualname__ = f"FnFormat_{spec.name}"
            self._fmt = _FnFormat()
        return self._fmt

    def __quack_semantic_key__(self):
        packed_key = None
        if self.packed is not None:
            packed_key = (
                self.packed.name,
                self.packed.w8,
                self.packed.tile_k,
                # make_consts shapes device code; the other host callables
                # don't reach the kernel (repack consistency is pinned by the
                # roundtrip fixture, not the cache key).
                _function_semantic_key(self.packed.make_consts)
                if self.packed.make_consts is not None
                else None,
            )
        return (
            "a_transform",
            _semantic_value_key(self.fn, set()),
            self.vec_size,
            _function_semantic_key(self.consts) if self.consts is not None else None,
            self.regs,
            packed_key,
            self.args,
        )


def _normalize_args(fn, args) -> tuple:
    """Validate an ``args`` declaration ({fn param name: kind name}) against
    the fn signature and the kind registry; normalize to ((name, kind), ...)
    in fn-parameter order (the staging order in the kernel)."""
    if not args:
        return ()
    from quack.operand_transform.transform_a import A_TRANSFORM_ARG_KINDS

    unknown = set(args.values()) - set(A_TRANSFORM_ARG_KINDS)
    assert not unknown, f"unknown operand kind(s) {unknown}; have {set(A_TRANSFORM_ARG_KINDS)}"
    params = list(inspect.signature(fn).parameters)
    missing = set(args) - set(params[1:])
    assert not missing, f"args {missing} are not parameters of {fn.__name__} (after x)"
    return tuple((name, args[name]) for name in params[1:] if name in args)


def _digest(key) -> str:
    return hashlib.sha256(repr(key).encode()).hexdigest()


class PackedFormatMod:
    """``transform_a=`` handle for a DecodeFormat: runs a registered (or
    instance) packed dequant format, e.g.
    ``GemmSm90(transform_a=w4_transform("qtip2s"))``."""

    def __init__(self, fmt):
        self.fmt = decode_format(fmt)
        self.packed = self.fmt
        self.name = f"w4_{self.fmt.name}"
        self.semantic_digest = _digest(
            (
                "w4_format",
                self.fmt.name,
                _function_semantic_key(type(self.fmt).decode_k16),
                _function_semantic_key(type(self.fmt).make_consts),
            )
        )

    def __call__(self, gemm):
        return TransformAW4(gemm, self.fmt)

    def __quack_semantic_key__(self):
        return ("w4_format_mod", self.semantic_digest)


def w4_transform(fmt) -> PackedFormatMod:
    """A ``transform_a=`` handle for a packed dequant format (name from the
    W4_FORMATS registry, or a DecodeFormat instance)."""
    return PackedFormatMod(fmt)


class DropoutAMod:
    """``transform_a=`` handle for dropout on A (see TransformADropout): the
    keep-mask of element (m, k) is a pure function of (m, k, seed, offset),
    reproducible by any kernel and invariant under split-k. MASK-ONLY — fold
    1/(1-p) into the epilogue. The (2,) int64 [seed, offset] CUDA tensor is
    the runtime operand, riding the TransformAOperand bundle's sf slot
    (``args`` declares it so the generic host plumbing — mod.gemm unpack,
    trace fakes, plan keys — treats it like any strip operand)."""

    packed = None
    consts = None
    regs = None
    args = (("seed", "seed_i64x2"),)

    def __init__(self, p: float, rounds: int = 7):
        assert 0.0 <= p < 1.0, "drop probability must be in [0, 1)"
        # keep iff byte >= threshold: P(drop) = threshold / 256, exactly
        self.p = p
        self.threshold = min(int(round(p * 256)), 255)
        self.rounds = rounds
        self.name = f"dropout_a_t{self.threshold}"
        # trailing int: scheme version
        self.semantic_digest = _digest(("dropout_a", self.threshold, rounds, 1))

    def __call__(self, gemm):
        from quack.operand_transform.transform_a import TransformADropout

        return TransformADropout(gemm, self)

    def __quack_semantic_key__(self):
        return ("dropout_a_mod", self.semantic_digest)


def dropout_a(p: float, rounds: int = 7) -> DropoutAMod:
    """A ``transform_a=`` handle for dropout on A with drop probability
    ``round(p * 256) / 256`` (mask-only; scale via the epilogue)."""
    return DropoutAMod(p, rounds)


def a_transform(
    vec_size: Optional[int] = None,
    packed=None,
    consts: Optional[Callable] = None,
    regs: Optional[tuple] = None,
    args: Optional[dict] = None,
):
    """Decorator: turn a plain fn into an A-operand transform mod. See the
    module docstring for the two fn contracts.

    ``consts=callable`` (value fns): called once per kernel (hoisted — LUTs,
    packed constants); its result is the fn's LAST parameter.
    ``args={param: kind}`` (value fns): runtime operands — each named fn
    parameter (between x and consts) receives its per-element values as a
    TensorSSA vector, staged by its kind (transform_a.A_TRANSFORM_ARG_KINDS;
    the (M, K) mirror of EpiOps' operand kinds). A crosses as the
    ``host.transform_a_operand(mod, A, {param: tensor}, tile_m)`` bundle.
    ``regs=(load, mma)`` overrides the register budget split (multiples of 8,
    see setmaxnreg constraints in TransformAW4)."""

    def wrap(fn):
        return ATransformMod(
            fn, vec_size=vec_size, packed=packed, consts=consts, regs=regs, args=args
        )

    return wrap
