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
(:class:`TransformAOperand`), not extra kernel parameters. Shipped: the
strip family at 2-D (gran_m, gran_k) granularity — ``colvec_ktile`` /
``colvec_k64/k32/k16`` (per-(row, k-group), e.g. the linear-CE dx pow2
rescale) and ``kvec_m64`` (per-(m64 block, k-element), the LCE dw strip);
plus dropout (:class:`TransformADropout` — seed rides the bundle RAW via
``aux_raw``, per-tile coordinates via the ``on_work_tile`` hook and the
seam's global ``k_tile``). There is deliberately no k-invariant colvec kind
(per-row scales commute to the epilogue). NOT ported yet: the fp8 m-major
layout transform.
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
      - ``make_copy_block(tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux,
        mAux)``: called in-kernel by each MMA warpgroup; returns
        ``copy_block(stage_idx, b, k_tile)`` which produces k16 block ``b``
        (a static Python int: register indexing) of pipeline stage
        ``stage_idx`` into the fragment ``tCrA``; ``k_tile`` is the GLOBAL
        k-tile index the block belongs to (split-k correct — needed only by
        coordinate-dependent transforms like dropout). The mainloop calls it
        under the rs_warpspecialized schedule (produce of block b+1 between
        WGMMA(b) and WGMMA(b+1), slot-0 preload of the next k-tile during
        the last block) — per-block work only; the schedule is never the
        transform's.
      - ``aux_raw``: the bundle's ``sf`` tensor is NOT an AuxOperandA TMA
        operand but a small raw gmem tensor (e.g. a dropout seed) handed to
        ``make_copy_block`` as ``mAux`` untouched — no smem, no pipeline.
      - ``uses_work_tile``: the mainloop calls ``on_work_tile(tile_coord_mnkl)``
        at each work-tile start (MMA warps, before the k-tile loop) so the
        transform can refresh per-tile register state (e.g. per-row RNG
        coordinates). ``mma_rs_interleaved`` runs once per work tile, so all
        ``copy_block`` calls between two hooks — including the next-k-tile
        slot-0 preload — belong to the hooked tile.
    """

    a_major_mode = cute.nvgpu.OperandMajorMode.K
    tile_k = None  # None -> kernel default
    owns_a_layout = False
    aux = None
    aux_raw = False
    uses_work_tile = False


class AuxKTileStrip(AuxOperandA):
    """Byte-granular per-(row-block, k-tile) aux strip: ``sf_bytes`` per m64
    block per k-tile, plain (sfb, tm64, stage) smem, one (sfb, tm64) box per
    k-tile arriving with A under the AB mbarrier. Semantically this carries
    a colvec-per-k-chunk operand in a PACKED encoding — W4's SF words are
    such an instance (repack-ordered per m64 atom for the pair_slot LDS,
    consumed inside decode_k16). The dense canonical form of the same
    concept is :class:`_StripAux` (element-typed boxes, no m64 structure
    needed)."""

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
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
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

        def copy_block(stage_idx, b, k_tile=None):
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
# the strip family — one value per (m-group, k-group) at 2-D granularity
# (gran_m, gran_k): the ``colvec_*`` corners (gran_m = 1: per-row, k at
# tile_K / 64 / 32 / 16) and ``kvec_m64`` (gran_m = 64, gran_k = 1: per
# (m64 block, k-element) — the LCE dw strip). Planned: ``colvec`` (per-row,
# k-invariant — needs the prologue hook), scalars/seeds.


def _strip_geometry(gemm, gran_m, gran_k):
    """(gran_m, g_m, gran_k, g_k, k_inner): resolved LAZILY — tile_K is 0 at
    transform-ctor time when the ctor tile shape is (M, N) (resolved in
    _setup_tiled_mma); every consumer below runs after that. ``k_inner``:
    the box's stride-1 axis is the FINER one — TMA needs 16 B alignment on
    every non-inner stride, and only the finer axis's group count is large
    enough to guarantee it (e.g. the dw strip's per-m-tile stride is
    tile_M/64 elements = 4-16 B)."""
    tile_m, tile_k = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[2]
    gran_m = tile_m if gran_m is None else gran_m
    gran_k = tile_k if gran_k is None else gran_k
    assert tile_m % gran_m == 0, f"gran_m {gran_m} must divide tile_M ({tile_m})"
    assert tile_k % gran_k == 0, f"gran_k {gran_k} must divide tile_K ({tile_k})"
    return gran_m, tile_m // gran_m, gran_k, tile_k // gran_k, gran_k < gran_m


class _StripAux(AuxOperandA):
    """Dense per-(m-group, k-group) values riding the aux-operand slot: a
    (g_m x g_k)-element MMA-dtype box per k-tile TMA'd into per-stage smem
    under the AB mbarrier — the values arrive WITH A, no gather latency in
    the produce path. Element-typed, so no byte/m64 factorization is needed
    (TMA box dims count ELEMENTS, <= 256 each: g_m <= tile_M <= 256 and
    g_k <= tile_K always hold). Box modes are (inner, outer) with the finer
    axis inner (see _strip_geometry); gmem view: (inner, outer, G_m, RestK,
    L) over a contiguous (outer-groups, inner-groups) tensor (see
    :func:`quack.operand_transform.host.transform_a_operand`)."""

    multicast = False  # small boxes: dup loads beat mcast box-splitting

    def __init__(self, gemm, gran_m, gran_k):
        self.gemm = gemm
        self.gran_m, self.gran_k = gran_m, gran_k
        self.dtype = gemm.mma_a_dtype

    def _box(self):
        _, g_m, _, g_k, k_inner = _strip_geometry(self.gemm, self.gran_m, self.gran_k)
        return (g_k, g_m) if k_inner else (g_m, g_k)

    def bytes_per_stage(self):
        box = self._box()
        return box[0] * box[1] * self.dtype.width // 8

    def make_smem_layout_staged(self, ab_stage):
        return cute.make_ordered_layout((*self._box(), ab_stage), order=(0, 1, 2))

    def make_tma(self, mAux):
        gemm = self.gemm
        box = self._box()
        smem_layout = cute.make_ordered_layout(box, order=(0, 1))
        return gemm._make_tma_atoms_and_tensors(mAux, smem_layout, box, gemm.cluster_shape_mnk[1])

    def gmem_slice(self, mAux, tile_coord_mnkl, batch_idx):
        # (inner, outer, Gm, RestK, L) -> (inner, outer, RestK)
        return mAux[None, None, tile_coord_mnkl[0], None, batch_idx]


class _StripArg:
    """The strip family: one MMA-dtype value per (m-group of ``gran_m`` A
    rows, k-group of ``gran_k`` elements), refreshed per k-tile. Corners:
    (1, tile_K) = ``colvec_ktile`` (the LCE dx pow2 rescale), (1, 16/32/64)
    = dense blockscaled-SF granularities, (64, 1) = ``kvec_m64`` (the LCE
    dw strip: per (vocab m64 block, token)). ``None`` means the whole tile
    extent on that axis.

    Staging (the epi_ops VecLoad idiom, all partition algebra — no index
    math): broadcast the smem box to (tile_M, tile_K) with NESTED modes —
    m-mode (gran_m, g_m) and k-mode (gran_k, g_k), stride 0 on the inner
    (within-group) levels — partition it with the fragment's own tiled_mma,
    and cache a fragment-congruent rmem tensor whose zero-stride modes share
    registers, refreshed once per k-tile with one LDS per distinct value
    (filter_zeros). Per-element reads are identity indexing, so the staging
    is selects only and the fn math stays packed (HMUL2). Which fragment
    slots share a value falls out of the layout composition — any
    granularities dividing the tile work, including quad-varying ones."""

    def __init__(self, gemm, gran_m, gran_k):
        self.gemm = gemm
        self.gran_m, self.gran_k = gran_m, gran_k  # geometry resolves lazily
        self.aux = _StripAux(gemm, gran_m, gran_k)
        self._tCsS = None
        self._rvals = None

    @cute.jit
    def setup(self, tiled_mma, tidx, mma_m, sAux):
        """Once per kernel (inside make_copy_block)."""
        assert sAux is not None, "strip operands ride the aux slot (pass A as TransformAOperand)"
        gemm = self.gemm
        gran_m, g_m, gran_k, g_k, k_inner = _strip_geometry(gemm, self.gran_m, self.gran_k)
        # Broadcast the (inner, outer, stage) box to (tile_M, tile_K, stage):
        # each axis is a nested (gran, g) mode — expand within a group
        # (stride 0), advance one box column across groups (the g factor is
        # the mode's second level, not a separate axis).
        sm, sk = (g_k, 1) if k_inner else (1, g_m)  # box strides of (m-group, k-group)
        sMK = cute.make_tensor(
            sAux.iterator,
            cute.make_layout(
                ((gran_m, g_m), (gran_k, g_k), gemm.ab_stage),
                stride=((0, sm), (0, sk), g_m * g_k),
            ),
        )
        # Partition with the fragment's own tiled_mma (epi_ops VecLoad
        # idiom): every value lands aligned with its fragment element — no
        # coordinate math. True lane: the fragment tensors are partitioned
        # from the per-warpgroup slice, but addressing needs the real thread.
        self._tCsS = tiled_mma.get_slice(tidx).partition_A(sMK)
        # fragment-congruent cache: make_rmem_tensor keeps the zero strides
        # (duplicates share a register) and compacts the rest, so the cache
        # holds exactly the per-lane distinct values
        self._rvals = cute.make_rmem_tensor(
            self._tCsS[None, None, None, 0].layout, gemm.mma_a_dtype
        )

    @cute.jit
    def on_block(self, stage_idx, b, mma_m):
        """Refresh the register cache — one LDS per DISTINCT value
        (filter_zeros pairs the deduped elements). k-coarse strips (a value
        spans >= one k16 block) load the whole tile's values once at b == 0;
        k-fine strips load block b's disjoint slice each block — the same
        produce rhythm as A itself, so the LDS spread across the WGMMA
        shadow and live ranges stay one block long (ptxas schedules within
        the unrolled body but won't restructure a whole-tile live range)."""
        gran_k = _strip_geometry(self.gemm, self.gran_m, self.gran_k)[2]
        if const_expr(gran_k >= 16):
            if const_expr(b == 0):
                cute.autovec_copy(
                    cute.filter_zeros(self._tCsS[None, None, None, stage_idx]),
                    cute.filter_zeros(self._rvals),
                )
        else:
            cute.autovec_copy(
                cute.filter_zeros(self._tCsS[None, None, b, stage_idx]),
                cute.filter_zeros(self._rvals[None, None, b]),
            )

    def element(self, coord, m, b):
        """The operand value of fragment element (coord, m, b): the cache is
        fragment-congruent, so this is identity indexing — the zero-stride
        modes resolve duplicates to the same register."""
        return self._rvals[coord, m, b]


# kind name -> kernel-side impl factory (host geometry: operand_transform.host)
A_TRANSFORM_ARG_KINDS = {
    "colvec_ktile": lambda gemm: _StripArg(gemm, gran_m=1, gran_k=None),
    "colvec_k64": lambda gemm: _StripArg(gemm, gran_m=1, gran_k=64),
    "colvec_k32": lambda gemm: _StripArg(gemm, gran_m=1, gran_k=32),
    "colvec_k16": lambda gemm: _StripArg(gemm, gran_m=1, gran_k=16),
    "kvec_m64": lambda gemm: _StripArg(gemm, gran_m=64, gran_k=1),
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
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
        consts = None
        if const_expr(self.mod.consts is not None):
            consts = self.mod.consts()  # hoisted: once per kernel
        load_block = canonical_a_load_s2r(tiled_mma, sA, tidx, tCrA, position_independent=True)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        for impl in self._arg_impls:
            impl.setup(tiled_mma, tidx, mma_m, sAux)

        def copy_block(stage_idx, b, k_tile=None):
            for impl in self._arg_impls:
                impl.on_block(stage_idx, b, mma_m)
            load_block(stage_idx, b)
            self._apply_block(tCrA, b, mma_m, consts)

        return copy_block


class TransformADropout(TransformA):
    """Dropout: a philox-derived keep-mask ANDed onto the fragment. MASK-ONLY
    — no 1/(1-p) multiply in the mainloop; fold the scale into the epilogue
    (an alpha, or an epi-mod multiply).

    Scheme (see quack/operand_transform/rng.py): the mask of element (m, k)
    is a pure function of (m, k, seed, offset) — one philox4x32 call per
    canonical group (row-pair x 32-k quad-strided set), which is exactly a
    lane's fragment ownership, so generation is thread-local (no shuffles,
    no redundancy) and any kernel (dgrad epilogue, wgrad) regenerates the
    same mask. Block parity p, row v1 consume philox word v1 + 2p whole; the
    per-register apply is PRMT (bytes -> constant-exponent b16 lanes) +
    set.ge.u32.b16x2 (0xFFFF lane mask) + AND: 3 SASS per 2 elements, no
    float math. The keep threshold int(round(p_drop * 256)) is a host
    constant baked into the mod's semantic key.

    Delivery: the (2,) int64 [seed, offset] tensor rides the
    :class:`TransformAOperand` bundle's sf slot RAW (``aux_raw`` — no
    TMA/smem); per-row coordinates refresh at each work tile via
    ``on_work_tile``; the mask is split-k invariant because the seam's
    ``k_tile`` is global (k_tile_start included)."""

    aux_raw = True
    uses_work_tile = True

    def __init__(self, gemm, mod):
        self.gemm = gemm
        self.mod = mod
        assert gemm.mma_a_dtype in (cutlass.BFloat16, cutlass.Float16), (
            "dropout masks 16-bit A fragments"
        )
        # tile_K may be unresolved (0) at ctor time for (M, N) ctor tile
        # shapes; geometry checks live in make_copy_block (lazy, like strips)

    @cute.jit
    def on_work_tile(self, tile_coord_mnkl):
        m_base = tile_coord_mnkl[0] * self.gemm.cta_tile_shape_mnk[0]
        for ma in cutlass.range_constexpr(self._mma_m):
            r0 = m_base + self._row0[ma]
            self._gm[ma] = (r0 // 16) * 8 + r0 % 8

    @cute.jit
    def _gen_pair(self, xw, span, mma_m):
        """philox words for global 32-k span ``span``, all m-atoms."""
        from cutlass import Uint64

        from quack.operand_transform import rng

        kg = (span << 2) | self._q
        for ma in cutlass.range_constexpr(mma_m):
            cnt = (Uint64(Int32(kg)) << Uint64(32)) | Uint64(self._gm[ma])
            x0, x1, x2, x3 = rng.philox(cnt, self._key, n_rounds=self.mod.rounds)
            xw[0, ma] = Int32(x0)
            xw[1, ma] = Int32(x1)
            xw[2, ma] = Int32(x2)
            xw[3, ma] = Int32(x3)

    @cute.jit
    def _mask_block(self, frag_i32, xw, b, mma_m):
        """AND the keep-mask onto k16 block b: block parity p, row v1 consume
        philox word v1 + 2p; per register one PRMT + SET + AND."""
        from quack.blockscaled.nvfp4_utils import prmt

        from quack.operand_transform import rng

        p = b % 2
        for ma in cutlass.range_constexpr(mma_m):
            for v1 in cutlass.range_constexpr(2):
                word = xw[v1 + 2 * p, ma]
                for h in cutlass.range_constexpr(2):
                    sel = rng.PRMT_SEL_H0 if h == 0 else rng.PRMT_SEL_H1
                    lanes = prmt(word, self._base_bytes, Int32(sel))
                    mask = rng.set_ge_u32_b16x2(Int32(lanes), self._thr_pair, self.gemm.mma_a_dtype)
                    frag_i32[(0, v1, h), ma, b] = frag_i32[(0, v1, h), ma, b] & mask

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx, sAux=None, mAux=None):
        from cutlass import Int64, Uint64

        from quack.operand_transform import rng

        gemm = self.gemm
        tile_m, tile_k = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[2]
        assert tile_k % 32 == 0, "dropout groups span 32 k"
        assert mAux is not None, "dropout needs the (2,) int64 [seed, offset] via mA.sf"
        load_block = canonical_a_load_s2r(tiled_mma, sA, tidx, tCrA, position_independent=True)
        base = rng.b16_base_pattern(gemm.mma_a_dtype)
        t = self.mod.threshold
        self._thr_pair = Int32((base | t) | ((base | t) << 16))
        self._base_bytes = Int32((base >> 8) * 0x01010101)
        # per-THREAD fragment coordinates: logical (m, k) of slot (e, v1, h)
        cA = cute.make_identity_tensor((tile_m, tile_k))
        tCcA = tiled_mma.get_slice(tidx).partition_A(cA)
        mma_m = const_expr(cute.size(tCcA.shape[1]))
        self._mma_m = mma_m
        # tile-relative row of each m-atom's first slot (static per lane) and
        # the quad class from the k coord of slot (e=0, v1=0, h=0): k = 2q
        self._row0 = [tCcA[(0, 0, 0), ma, 0][0] for ma in range(mma_m)]
        self._q = (tCcA[(0, 0, 0), 0, 0][1] % 8) // 2
        self._gm = cute.make_rmem_tensor((mma_m,), Int32)
        seed, offset = Int64(mAux[0]), Int64(mAux[1])
        # offset stream folded into the key (counter words carry coordinates)
        self._key = (seed + offset * Int64(rng.PHILOX_OFFSET_MIX)).to(Uint64)
        xw = cute.make_rmem_tensor((4, mma_m), Int32)
        frag_i32 = cute.recast_tensor(tCrA, Int32)
        spans = const_expr(tile_k // 32)

        def copy_block(stage_idx, b, k_tile):
            load_block(stage_idx, b, k_tile)
            # regenerate at each 32-k span boundary; span s+1 lands in the
            # WGMMA shadow like every other produce
            if const_expr(b % 2 == 0):
                self._gen_pair(xw, k_tile * spans + b // 2, mma_m)
            self._mask_block(frag_i32, xw, b, mma_m)

        return copy_block
