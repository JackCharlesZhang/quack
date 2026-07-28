# Copyright (c) 2026, Tri Dao.
"""A-operand transforms for the SM90 RS GEMM mainloop.

The RS mainloop (``GemmSm90.mma_rs_interleaved``, CUTLASS rs_warpspecialized
scheme) produces the WGMMA A fragment one k16 block at a time through an
abstract seam: ``copy_block(stage_idx, b)``. The default produce is the
canonical ldmatrix s2r load (``sm90_utils.canonical_a_load_s2r``); a
transform substitutes its own — a dequant of packed weights, or a value fn
applied on the way — while the mainloop keeps owning the WGMMA issue, the
commit-group discipline and the pipeline waits.

The kernel is agnostic to what the transform computes. It only consumes the
declarative contract below: A's storage layout (possibly transform-owned),
the required tile_K, and the in-kernel ``make_copy_block`` hook.

Ported from the transformA branch, restructured for the interleaved mainloop
(the branch's whole-tile ``make_mma_fn`` / double-buffered fragment scheme is
gone — main has a single tile-wide fragment and per-block produce). Runtime
operands of value fns are a KIND taxonomy over the transform's (M, K) index
space (see the fn-operand kinds section / A_TRANSFORM_ARG_KINDS), delivered
via the aux A-side TMA slot bundled into the mA argument
(:class:`TransformAOperand`), not extra kernel parameters. Shipped:
``colvec_ktile`` (per-(row, k-tile), e.g. the linear-CE dx pow2 rescale).
NOT ported yet: k-invariant colvec / dropout-seed operands (branch
mTransformAArg + per-work-tile prologue hook) and the fp8 m-major layout
transform.
"""

from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from quack.blockscaled.decode_formats import decode_format
from quack.cute_dsl_utils import mlir_namedtuple
from quack.sm90_utils import canonical_a_load_s2r


@mlir_namedtuple
class TransformAOperand(NamedTuple):
    """The A operand of a layout-owning transform, crossing the kernel
    boundary as ONE bundled argument in the mA slot — the mainloop analogue
    of EpilogueArguments: the host layer never learns the bundle's anatomy,
    and the GEMM signature arity stays fixed for plain GEMMs. ``blob`` is
    the repacked storage; ``sf`` the optional aux strip (TMA'd per k-tile
    alongside A under the same mbarrier — see AuxOperandA). Future transform
    runtime operands (colvec scales, dropout seeds) become new optional
    fields here, not new kernel parameters."""

    blob: cute.Tensor
    sf: Optional[cute.Tensor] = None


class AuxOperandA:
    """An extra A-side operand riding the AB pipeline: one TMA box per k-tile
    into its own per-stage smem buffer, arriving under the same mbarrier as A
    and B. GemmSm90 consumes this protocol (duck-typed); anything — a
    transform, or a future standalone feature — can install one.

    Contract:
      - ``dtype``: cutlass numeric type of the smem buffer.
      - ``bytes_per_stage()``: smem bytes per pipeline stage (stage-count
        heuristic input; must match the staged layout below).
      - ``make_smem_layout_staged(ab_stage)``: the (…, ab_stage) smem layout.
      - ``make_tma(mAux)``: TMA atom + tma tensor for the gmem operand.
      - ``gmem_slice(mAux, tile_coord_mnkl, batch_idx)``: the per-CTA-tile
        gmem view whose last mode walks k-tiles (source for the block copy).
      - ``multicast`` (optional, default True): False opts out of the A-side
        cluster multicast — small boxes (e.g. 128 B scale strips) load a full
        copy per CTA instead of splitting the box.
    """

    dtype = cutlass.Uint8


class TransformA:
    """A-operand transform: produce the WGMMA A fragment from smem in
    registers each k16 block, instead of the canonical ldmatrix s2r load.

    The contract with GemmSm90 is declarative — the kernel never learns what
    the transform computes, only:
      - The MMA compute dtype is the GEMM's own ``mma_a_dtype`` (its a_dtype
        constructor arg) — a transform never changes it, it must PRODUCE
        fragments in it (validate support in ``__init__``). ``a_major_mode``:
        the fragment major declared by layout-owning transforms (a storage
        blob has no natural major; must be canonical in K — B stays SS behind
        a canonical descriptor, so no K reorder is absorbable). ``tile_k``:
        required tile_K, or None.
      - ``owns_a_layout``: mA is not an (M, K) operand (e.g. a repacked blob);
        the transform then owns A's smem layout (``make_a_smem_layout_staged``,
        ``a_bytes_per_stage``), TMA (``make_a_tma``) and gmem slicing
        (``a_gmem_slice``), and the kernel skips (M, K)-based checks, batch
        rotation and length derivation (M comes from D).
      - ``aux``: optional :class:`AuxOperandA` installed by this transform;
        its smem arrives in ``make_copy_block`` as ``sAux``. The aux facility
        is transform-agnostic (per-row scales for a plain bf16 GEMM could
        ride it without any transform); a transform merely *installs* one
        (W4's scale-factor strip).
      - ``__init__(gemm)`` validates the config and may adjust register
        budgets / occupancy (runs after the gemm's defaults, before
        _setup_attributes).
      - ``make_copy_block(tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux)``:
        called in-kernel by each MMA warpgroup; returns ``copy_block(stage_idx, b)``
        which produces k16 block ``b`` (a static Python int: register
        indexing) of pipeline stage ``stage_idx`` into the fragment ``tCrA``.
        The mainloop calls it under the rs_warpspecialized schedule (produce
        of block b+1 between WGMMA(b) and WGMMA(b+1), slot-0 preload of the
        next tile during the last block) — per-block work only; the schedule
        is never the transform's.
    """

    a_major_mode = cute.nvgpu.OperandMajorMode.K
    tile_k = None  # None -> kernel default
    owns_a_layout = False
    aux = None


class AuxKTileStrip(AuxOperandA):
    """Byte-granular per-(row-block, k-tile) aux strip: ``sf_bytes`` per m64
    block per k-tile, plain (sfb, tm64, stage) smem, one (sfb, tm64) box per
    k-tile arriving with A under the AB mbarrier. Semantically this carries
    a colvec-per-k-chunk operand in a PACKED encoding — W4's SF words are
    such an instance (repack-ordered per m64 atom for the pair_slot LDS,
    consumed inside decode_k16). The dense canonical form of the same
    concept is :class:`_ColVecStripAux` (element-typed (tile_M, g) boxes, no
    m64 structure needed)."""

    def __init__(self, gemm, sf_bytes):
        self.gemm = gemm
        self.sf_bytes = sf_bytes

    def _tm64(self):
        return self.gemm.cta_tile_shape_mnk[0] // 64

    def bytes_per_stage(self):
        return self.sf_bytes * self._tm64()

    def make_smem_layout_staged(self, ab_stage):
        return cute.make_ordered_layout((self.sf_bytes, self._tm64(), ab_stage), order=(0, 1, 2))

    def make_tma(self, mAux):
        gemm = self.gemm
        sf_smem_layout = cute.make_ordered_layout((self.sf_bytes, self._tm64()), order=(0, 1))
        return gemm._make_tma_atoms_and_tensors(
            mAux, sf_smem_layout, (self.sf_bytes, self._tm64()), gemm.cluster_shape_mnk[1]
        )

    def gmem_slice(self, mAux, tile_coord_mnkl, batch_idx):
        # (sfb, tm64, Gt, RestK, L) -> (sfb, tm64, RestK)
        return mAux[None, None, tile_coord_mnkl[0], None, batch_idx]


class TransformAW4(TransformA):
    """Packed 4-bit / 8-bit weights as operand A, decoded to bf16 in
    registers and fed to RS WGMMA (W4A16; Hopper has no fp4 tensor cores).

    mA is the offline-repacked blob (see blockscaled/nvfp4_utils.repack_*):
    per (m64, k-tile) block each thread's 16 B (32 B for 8-bit weights or
    tile_k=128 formats) LDS lands values directly in WGMMA A-fragment order,
    so the decode is shuffle-free. ``copy_block(stage, 0)`` LDSes the k-tile's
    raw words and decodes block 0; blocks 1.. decode from the same registers
    (a tile's raw words are dead before the next tile's slot-0 produce, so a
    single register set suffices).

    Scale factors ride the aux-operand slot (a per-stage strip TMA'd next to
    A under the same mbarrier). Formats: nvfp4 (e2m1 + e4m3 SF per 16, scale
    folded in the decode — exact), int4 (u4b8 + bf16 group scale), int4awq
    (scale + zero, one HFMA2), mxfp4/mxfp8 (e8m0 per 32), int8/fp8 (no strip;
    per-channel scale left to the epilogue), qtip* (no strip; per-tensor
    scale rides alpha).

    This transform is requested explicitly rather than layout-detected: mA's
    shape alone does not identify the format. D is typically written
    (N_w, M_act) m-major (out = act @ W^T row-major).
    """

    owns_a_layout = True

    def __init__(self, gemm, w4_format):
        self.fmt = decode_format(w4_format)
        assert gemm.mma_a_dtype == cutlass.BFloat16, "w4 decodes to bf16 (W4A16)"
        self.gemm = gemm
        self.w4_format = self.fmt.name
        self.tile_k = self.fmt.tile_k
        # raw i32 words per thread per m64 block per stage
        self._nw = (8 if self.fmt.w8 else 4) * (self.tile_k // 64)
        # split_k measured working + winning on grid-starved decode shapes
        # (N/tile_m CTAs < machine): serial split-k 1.2-1.5x there.
        assert not gemm.gather_A
        assert not gemm.pingpong, "w4 only supports cooperative for now"
        assert gemm.atom_layout_mnk[1] == 1, "w4 requires atom_layout_n == 1"
        assert gemm.cta_tile_shape_mnk[0] % 64 == 0, "w4 requires tile_M % 64 == 0"
        assert gemm.cluster_shape_mnk[0] == 1 and gemm.cluster_shape_mnk[2] == 1, (
            "w4 supports (1, cluster_N, 1) clusters"
        )
        if self.fmt.sf_words > 0:
            # the format's SF words: a compressed colvec-per-k-group instance
            # of the k-tile strip geometry, consumed inside decode_k16
            self.aux = AuxKTileStrip(gemm, self.fmt.sf_bytes)
        if const_expr(gemm.cta_tile_shape_mnk[1] <= 32):
            # Small-N (decode) shapes are latency-bound: consumers need few
            # regs (small acc + A frag), so shrink budgets to fit 2 CTAs/SM
            # and double the warps available to hide LDS/decode latency.
            # Budget: with min_blocks_per_mp=2 ptxas caps launch regs at
            # floor(65536 / (2*threads) / 8) * 8, and setmaxnreg deadlocks if
            # the inc demand exceeds what the producer's dec released, so
            # keep 128*load + 256*mma (2 WG) within threads*launch_regs.
            gemm.occupancy = 2
            if gemm.mma_warp_groups == 2:
                gemm.num_regs_load, gemm.num_regs_mma = 32, 104  # 384thr @ 80
            else:
                # 2 WGs @ 256 threads, launch cap 128: math can take up to
                # (256*128 - 128*40)/128 = 216; give it slack so ptxas
                # keeps decode LUT constants resident instead of UR->R
                # rematerializing them in the mainloop.
                gemm.num_regs_load, gemm.num_regs_mma = 40, 152

    # ---- A layout ownership -------------------------------------------------

    def a_bytes_per_stage(self):
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        return 256 * (2 * self._nw) * tm64

    def make_a_smem_layout_staged(self, ab_stage):
        """A smem holds the repacked blob, 4 * nw B per thread slot per m64
        block, no swizzle; TMA-facing shape has a 256 B inner run."""
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        return cute.make_ordered_layout(
            (256, 2 * self._nw, tm64, ab_stage),
            order=(0, 1, 2, 3),
        )

    def make_a_tma(self, mA):
        """mA is the blob (256, 2*nw, tm64, Gt, Kt, L); one (256, 2*nw, tm64)
        box per k-tile."""
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        araw_smem_layout = cute.slice_(gemm.a_smem_layout_staged, (None, None, None, 0))
        return gemm._make_tma_atoms_and_tensors(
            mA,
            araw_smem_layout,
            (256, 2 * self._nw, tm64),
            gemm.cluster_shape_mnk[1],
        )

    def a_gmem_slice(self, mA, tile_coord_mnkl, batch_idx):
        # (256, 8|16, tm64, Gt, RestK, L) -> (256, 8|16, tm64, RestK)
        return mA[None, None, None, tile_coord_mnkl[0], None, batch_idx]

    # ---- the per-block produce ----------------------------------------------

    @cute.jit
    def _decode_block(self, xw, sfw, frag_i32, b, mma_m, consts):
        """Decode k16 block b (all MMA_M atoms) from preloaded raw words: the
        format's decode_k16 produces the 4 packed bf16x2 registers per m-atom
        in fragment slot order; the slot assignment here is format-agnostic."""
        for m in cutlass.range_constexpr(mma_m):
            r0, r1, r2, r3 = self.fmt.decode_k16(xw[None, m], sfw[None, m], b, consts)
            frag_i32[(0, 0, 0), m, b] = r0
            frag_i32[(0, 1, 0), m, b] = r1
            frag_i32[(0, 0, 1), m, b] = r2
            frag_i32[(0, 1, 1), m, b] = r3

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None):
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        nw = self._nw
        sA_i32 = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=Int32),
            cute.make_ordered_layout((nw, 128, tm64, gemm.ab_stage), order=(0, 1, 2, 3)),
        )
        sAux_i32 = cute.recast_tensor(sAux, Int32) if const_expr(sAux is not None) else None
        t128 = tidx % 128
        # this thread's SF word slot within the m64 block's 32 fragment rows:
        # (warp, quad) -> row pair (matches repack_*_sf's word order)
        pair_slot = (t128 // 32) * 8 + (t128 % 32) // 4
        consts = self.fmt.make_consts()
        frag_i32 = cute.recast_tensor(tCrA, Int32)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        atom_m = gemm.atom_layout_mnk[0]
        sf_words = self.fmt.sf_words
        xw = cute.make_rmem_tensor((nw, mma_m), Int32)
        sfw = cute.make_rmem_tensor((2, mma_m), Int32)

        def copy_block(stage_idx, b):
            if const_expr(b == 0):
                for m in cutlass.range_constexpr(mma_m):
                    m64 = m * atom_m + warp_group_idx
                    cute.autovec_copy(sA_i32[None, t128, m64, stage_idx], xw[None, m])
                    for w in cutlass.range_constexpr(sf_words):
                        sfw[w, m] = sAux_i32[sf_words * pair_slot + w, m64, stage_idx]
            self._decode_block(xw, sfw, frag_i32, b, mma_m, consts)

        return copy_block


# ---- fn-operand kinds --------------------------------------------------------
#
# Runtime operands of value-fn transforms, a taxonomy over the transform's
# (M, K) index space — the mainloop mirror of EpiOps' operand kinds over
# (M, N) (scalar / colvec / rowvec / tile). Each kind owns its indexing
# (which fragment coordinate the value depends on), its delivery (aux TMA
# box geometry vs a future per-work-tile prologue load), and its register
# staging; the fn only ever sees a vec of values aligned with x. Shipped:
# the ``colvec_*`` strip family (per-(row, k-group) values at granularity
# tile_K / 64 / 32 / 16 elements). Planned: ``colvec`` (per-row, k-invariant
# — needs the prologue hook), ``kvec`` (per-k-element, the LCE dw strip),
# scalars/seeds.


def _strip_geometry(gemm, granularity):
    """(granularity, g): resolved LAZILY — tile_K is 0 at transform-ctor time
    when the ctor tile shape is (M, N) (resolved in _setup_tiled_mma); every
    consumer below runs after that."""
    tile_k = gemm.cta_tile_shape_mnk[2]
    gran = tile_k if granularity is None else granularity
    assert gran % 8 == 0 and tile_k % gran == 0, (
        f"granularity {gran} must be a multiple of 8 dividing tile_K ({tile_k})"
    )
    return gran, tile_k // gran


class _ColVecStripAux(AuxOperandA):
    """Dense per-(A-row, k-group) values riding the aux-operand slot: a
    (tile_M, g) MMA-dtype box per k-tile (g values per row per k-tile) TMA'd
    into per-stage smem under the AB mbarrier — the values arrive WITH A, no
    gather latency in the produce path. Element-typed, so no byte/m64
    factorization is needed (TMA box dims count ELEMENTS, <= 256 each:
    tile_M <= 256 always holds). Gmem view: (tile_M, g, G_m, RestK, L) over
    a contiguous (rk * g, M) tensor (see
    :func:`quack.operand_transform.host.transform_a_operand`)."""

    multicast = False  # small boxes: dup loads beat mcast box-splitting

    def __init__(self, gemm, granularity):
        self.gemm = gemm
        self.granularity = granularity
        self.dtype = gemm.mma_a_dtype

    def _tile_m_g(self):
        return self.gemm.cta_tile_shape_mnk[0], _strip_geometry(self.gemm, self.granularity)[1]

    def bytes_per_stage(self):
        tile_m, g = self._tile_m_g()
        return tile_m * g * self.dtype.width // 8

    def make_smem_layout_staged(self, ab_stage):
        tile_m, g = self._tile_m_g()
        return cute.make_ordered_layout((tile_m, g, ab_stage), order=(0, 1, 2))

    def make_tma(self, mAux):
        gemm = self.gemm
        tile_m, g = self._tile_m_g()
        smem_layout = cute.make_ordered_layout((tile_m, g), order=(0, 1))
        return gemm._make_tma_atoms_and_tensors(
            mAux, smem_layout, (tile_m, g), gemm.cluster_shape_mnk[1]
        )

    def gmem_slice(self, mAux, tile_coord_mnkl, batch_idx):
        # (tile_M, g, Gm, RestK, L) -> (tile_M, g, RestK)
        return mAux[None, None, tile_coord_mnkl[0], None, batch_idx]


class _ColVecStripArg:
    """The ``colvec_*`` strip family: one MMA-dtype value per (A row,
    k-group of ``granularity`` elements) — colvec-indexed like EpiOps'
    colvec (per M), refreshed per k-tile with g = tile_K / granularity
    values per row. ``granularity=None`` means tile_K (g = 1, the
    ``colvec_ktile`` kind — the LCE dx pow2 rescale).

    Staging: a (g, 2, MMA_M) register cache refreshed once per k-tile (a
    lane's fragment rows are k-invariant: the mode-0 row bit x MMA_M); the
    k-group of a fragment slot is (16*b + 8*k_half) // granularity — STATIC
    for granularity >= 8 (lane quads never straddle an 8-element boundary) —
    so the per-element staging is selects only and the fn math stays packed
    (HMUL2)."""

    def __init__(self, gemm, granularity=None):
        self.gemm = gemm
        self.granularity = granularity  # geometry resolves lazily (tile_K)
        self.aux = _ColVecStripAux(gemm, granularity)
        self._saux = None
        self._rows = None
        self._rvals = None

    @cute.jit
    def setup(self, tiled_mma, tidx, mma_m, sAux):
        """Once per kernel (inside make_copy_block)."""
        assert sAux is not None, "colvec strips ride the aux slot (pass A as TransformAOperand)"
        gemm = self.gemm
        self._gran, self._g = _strip_geometry(gemm, self.granularity)
        self._saux = sAux  # (tile_M, g, stage), already in the MMA dtype
        # This lane's fragment row coordinates (tile-local, k-invariant:
        # the row bit x MMA_M). The fragment tensors are partitioned from
        # the per-warpgroup slice; coordinates need the true lane.
        cA = cute.make_identity_tensor((gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[2]))
        tCcA = tiled_mma.get_slice(tidx).partition_A(cA)
        self._rows = cute.make_rmem_tensor((2, mma_m), Int32)
        for v1 in cutlass.range_constexpr(2):
            for mi in cutlass.range_constexpr(mma_m):
                self._rows[v1, mi] = tCcA[(0, v1, 0), mi, 0][0]
        self._rvals = cute.make_rmem_tensor((self._g, 2, mma_m), gemm.mma_a_dtype)

    @cute.jit
    def on_ktile(self, stage_idx, mma_m):
        """At each k-tile's first block: refresh the per-(group, row) value
        cache from this stage's aux smem box."""
        for gi in cutlass.range_constexpr(self._g):
            for v1 in cutlass.range_constexpr(2):
                for mi in cutlass.range_constexpr(mma_m):
                    self._rvals[gi, v1, mi] = self._saux[self._rows[v1, mi], gi, stage_idx]

    def element(self, coord, m, b):
        """The operand value of fragment element (coord, m, b) — static
        indexing into the cache (the kind's coordinate dependence: the
        k-group of (block, k-half), the row bit coord[1], and the m-atom)."""
        return self._rvals[(16 * b + 8 * coord[2]) // self._gran, coord[1], m]


# kind name -> kernel-side impl factory (host geometry: operand_transform.host)
A_TRANSFORM_ARG_KINDS = {
    "colvec_ktile": lambda gemm: _ColVecStripArg(gemm, granularity=None),
    "colvec_k64": lambda gemm: _ColVecStripArg(gemm, granularity=64),
    "colvec_k32": lambda gemm: _ColVecStripArg(gemm, granularity=32),
    "colvec_k16": lambda gemm: _ColVecStripArg(gemm, granularity=16),
}


class TransformAValue(TransformA):
    """Value transform on an unpacked 16-bit A: the canonical ldmatrix s2r
    load, then the mod's fn applied in-place over the block's fragment
    elements in ``vec_size`` chunks (running in the WGMMA shadow under the
    interleaved schedule). The fn contract (see frontend.py): one lane's
    ``vec_size`` fragment elements as a TensorSSA vector in the MMA dtype,
    FRAGMENT-SLOT-ORDERED (2 adjacent k x 2 rows x 2 k-halves per block —
    not k-contiguous), same-length vector out; chunks are pair-aligned, so
    vec_size in {2, 4, 8}.

    ``mod.args`` ((param_name, kind) pairs): runtime operands — the fn's
    parameters between x and consts, each staged per element by its kind
    (see the fn-operand kinds section above / A_TRANSFORM_ARG_KINDS). At
    most one aux-delivered operand for now (the bundle has a single aux
    slot); the host passes A as a ``TransformAOperand(A, view)`` bundle
    built by :func:`quack.operand_transform.host.transform_a_operand`."""

    def __init__(self, gemm, mod):
        self.gemm = gemm
        self.mod = mod
        assert gemm.mma_a_dtype.width == 16, (
            "value transforms ride the canonical ldmatrix s2r load (16-bit only)"
        )
        if getattr(mod, "regs", None) is not None:
            gemm.num_regs_load, gemm.num_regs_mma = mod.regs
        self._arg_impls = [
            A_TRANSFORM_ARG_KINDS[kind](gemm) for _name, kind in getattr(mod, "args", ()) or ()
        ]
        aux_impls = [impl for impl in self._arg_impls if getattr(impl, "aux", None) is not None]
        assert len(aux_impls) <= 1, "one aux-delivered operand per transform (single aux slot)"
        if aux_impls:
            self.aux = aux_impls[0].aux

    @cute.jit
    def _apply_block(self, buf, b, mma_m, consts):
        """Apply the fn to k16 block b, vec_size at a time. The scalar
        staging copies are register moves that fold away in SSA; element
        order i0-fastest keeps chunks pair-aligned."""
        vec = self.mod.vec_size
        s0 = buf.shape[0]  # ((2, 2, 2)) fragment slot mode
        coords = [
            (i2, i1, i0)
            for i0 in range(cute.size(s0[2]))
            for i1 in range(cute.size(s0[1]))
            for i2 in range(cute.size(s0[0]))
        ]
        for m in cutlass.range_constexpr(mma_m):
            for c in cutlass.range_constexpr(len(coords) // vec):
                tmp = cute.make_rmem_tensor((vec,), buf.element_type)
                for i in cutlass.range_constexpr(vec):
                    tmp[i] = buf[coords[c * vec + i], m, b]
                args = [tmp.load()]
                for impl in self._arg_impls:
                    # per-element operand values, staged from the kind's
                    # register cache (folds to selects; fragment dtype so
                    # the fn math stays packed)
                    sv = cute.make_rmem_tensor((vec,), buf.element_type)
                    for i in cutlass.range_constexpr(vec):
                        sv[i] = impl.element(coords[c * vec + i], m, b)
                    args.append(sv.load())
                if const_expr(self.mod.consts is not None):
                    args.append(consts)
                y = self.mod.fn(*args)
                if const_expr(y.dtype != buf.element_type):
                    # fn math may promote (e.g. TensorSSA * python float ->
                    # f32); convert back to the MMA dtype — packed cvt.rn,
                    # free (see memory: dsl-to-packed-cvt).
                    y = y.to(buf.element_type)
                tmp.store(y)
                for i in cutlass.range_constexpr(vec):
                    buf[coords[c * vec + i], m, b] = tmp[i]

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None):
        consts = None
        if const_expr(self.mod.consts is not None):
            consts = self.mod.consts()  # hoisted: once per kernel
        load_block = canonical_a_load_s2r(tiled_mma, sA, tidx, tCrA, position_independent=True)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        for impl in self._arg_impls:
            impl.setup(tiled_mma, tidx, mma_m, sAux)

        def copy_block(stage_idx, b):
            if const_expr(b == 0):
                for impl in self._arg_impls:
                    impl.on_ktile(stage_idx, mma_m)
            load_block(stage_idx, b)
            self._apply_block(tCrA, b, mma_m, consts)

        return copy_block
