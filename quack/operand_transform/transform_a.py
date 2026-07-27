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
gone — main has a single tile-wide fragment and per-block produce). NOT
ported yet: the aux A-side operand (per-stage TMA strip riding the AB
pipeline — W4's scale-factor strips need it), runtime operand ports
(mTransformAArg: colvec, dropout seeds), and the fp8 m-major layout
transform.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from quack.blockscaled.decode_formats import decode_format
from quack.sm90_utils import canonical_a_load_s2r


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
      - ``__init__(gemm)`` validates the config and may adjust register
        budgets / occupancy (runs after the gemm's defaults, before
        _setup_attributes).
      - ``make_copy_block(tiled_mma, sA, tCrA, tidx, warp_group_idx)``: called
        in-kernel by each MMA warpgroup; returns ``copy_block(stage_idx, b)``
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

    Formats without a scale-factor strip only, for now (qtip*, int8/fp8 with
    the per-channel scale left to the epilogue): the strip rides the aux
    A-side operand, which is not ported from the transformA branch yet.

    This transform is requested explicitly rather than layout-detected: mA's
    shape alone does not identify the format. D is typically written
    (N_w, M_act) m-major (out = act @ W^T row-major).
    """

    owns_a_layout = True

    def __init__(self, gemm, w4_format):
        self.fmt = decode_format(w4_format)
        assert gemm.mma_a_dtype == cutlass.BFloat16, "w4 decodes to bf16 (W4A16)"
        assert self.fmt.sf_words == 0, (
            f"format {self.fmt.name!r} needs a scale-factor strip; the aux A-side operand "
            "(SFA) is not ported from the transformA branch yet"
        )
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
    def _decode_block(self, xw, frag_i32, b, mma_m, consts):
        """Decode k16 block b (all MMA_M atoms) from preloaded raw words: the
        format's decode_k16 produces the 4 packed bf16x2 registers per m-atom
        in fragment slot order; the slot assignment here is format-agnostic."""
        for m in cutlass.range_constexpr(mma_m):
            r0, r1, r2, r3 = self.fmt.decode_k16(xw[None, m], None, b, consts)
            frag_i32[(0, 0, 0), m, b] = r0
            frag_i32[(0, 1, 0), m, b] = r1
            frag_i32[(0, 0, 1), m, b] = r2
            frag_i32[(0, 1, 1), m, b] = r3

    @cute.jit
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx):
        gemm = self.gemm
        tm64 = gemm.cta_tile_shape_mnk[0] // 64
        nw = self._nw
        sA_i32 = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=Int32),
            cute.make_ordered_layout((nw, 128, tm64, gemm.ab_stage), order=(0, 1, 2, 3)),
        )
        t128 = tidx % 128
        consts = self.fmt.make_consts()
        frag_i32 = cute.recast_tensor(tCrA, Int32)
        mma_m = const_expr(cute.size(tCrA.shape[1]))
        atom_m = gemm.atom_layout_mnk[0]
        xw = cute.make_rmem_tensor((nw, mma_m), Int32)

        def copy_block(stage_idx, b):
            if const_expr(b == 0):
                for m in cutlass.range_constexpr(mma_m):
                    m64 = m * atom_m + warp_group_idx
                    cute.autovec_copy(sA_i32[None, t128, m64, stage_idx], xw[None, m])
            self._decode_block(xw, frag_i32, b, mma_m, consts)

        return copy_block


class TransformAValue(TransformA):
    """Value transform on an unpacked 16-bit A: the canonical ldmatrix s2r
    load, then the mod's fn applied in-place over the block's fragment
    elements in ``vec_size`` chunks (running in the WGMMA shadow under the
    interleaved schedule). The fn contract (see frontend.py): one lane's
    ``vec_size`` fragment elements as a TensorSSA vector in the MMA dtype,
    FRAGMENT-SLOT-ORDERED (2 adjacent k x 2 rows x 2 k-halves per block —
    not k-contiguous), same-length vector out; chunks are pair-aligned, so
    vec_size in {2, 4, 8}."""

    def __init__(self, gemm, mod):
        self.gemm = gemm
        self.mod = mod
        assert gemm.mma_a_dtype.width == 16, (
            "value transforms ride the canonical ldmatrix s2r load (16-bit only)"
        )
        if getattr(mod, "regs", None) is not None:
            gemm.num_regs_load, gemm.num_regs_mma = mod.regs

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
    def make_copy_block(self, tiled_mma, sA, tCrA, tidx, warp_group_idx):
        consts = None
        if const_expr(self.mod.consts is not None):
            consts = self.mod.consts()  # hoisted: once per kernel
        load_block = canonical_a_load_s2r(tiled_mma, sA, tidx, tCrA, position_independent=True)
        mma_m = const_expr(cute.size(tCrA.shape[1]))

        def copy_block(stage_idx, b):
            load_block(stage_idx, b)
            self._apply_block(tCrA, b, mma_m, consts)

        return copy_block
