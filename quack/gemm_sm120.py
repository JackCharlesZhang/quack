# Copyright (c) 2025-2026, QuACK team.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py
# SM120-style GEMM using warp-level MMA (MmaF16BF16Op) + ldmatrix.
# Unlike SM90 WGMMA (which reads A/B from SMEM directly), warp-level MMA
# requires explicit SMEM→RMEM copies via ldmatrix before each MMA instruction.

# Measured facts (RTX 5090, 2026-07-29 — see AI/sm120_transform_fp8_tuning.md
# for the full session data; verify before assuming they changed):
# - Warp-mma dense rates (boosted clocks): bf16/f32acc 255 TF, fp8
#   kind::f8f6f4/f32acc 507 TF, fp8/f16acc 1017 TF, and kind::mxf8f6f4
#   (block-scaled) fp8 with FP32 acc at 1005 TF — the block-scaled
#   instruction is 2x the plain fp8 mma at the same accumulator, which is
#   why fp8 rides MmaMXF8Op with constant unit ue8m0 scales below.
# - fp8 mma.sync f32 accumulate keeps ~21-22 mantissa bits, TRUNCATING (RZ):
#   +1 onto 2^n survives to n=20 (bf16 datapath: n=21); identical boundary
#   for the mxf8f6f4 instruction. No Hopper-style slow accum needed; drift
#   is ~(K/32)*2^-21 relative.
# - W4 decode shapes (m <= 64) want tile_m=128 with split-k pushed until
#   k-tiles/split < 32 (relax to 16 under ~96 CTAs); the 170-SM part rewards
#   grids well past the H100 112-CTA target. Short-K prefill wants tile_n
#   128 (pick_w4_cfg's sm120 branch has the measured numbers).
# - PTX 9.2 direct fp4/fp8->bf16x2 cvts decode 2.0x/1.6x faster than the
#   sm90 prmt-LUT/f16-route sequences (2.45x with the fused ue8m0 scale) —
#   see quack/blockscaled/nvfp4_utils.py `_arch_has_bf16_narrow_cvt`.

import math
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warp
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.utils import SmemPartition

import cutlass.utils.blackwell_helpers as blackwell_helpers
from cutlass.utils import blockscaled_layout

from quack.varlen_utils import VarlenManager
from quack.pipeline import make_pipeline_state
from quack import copy_utils
from quack.gemm_sm90 import GemmSm90, NamedBarrierGemm
from quack.gemm_config import SplitKMode
from quack.tile_scheduler import ag_wait_m_tile
from quack import sm80_utils
import quack.sm90_utils as quack_sm90_utils


def _sf_group_vmk(t, k_atoms):
    """Group a raw SM120 SF fragment (V, rest-MN modes..., K modes...) to
    rank-3 (V, MN, K), walking the K modes off the right until their sizes
    multiply to ``k_atoms`` (= tile_K / 32). Plain-python trace-time helper:
    shapes are static, and it must NOT run under the DSL preprocessor (an
    in-kernel ``while`` is rewritten to dynamic control flow, turning the
    mode index into an Int32 that cute.size rejects)."""
    r = cute.rank(t)
    i, prod = r, 1
    while prod < k_atoms and i > 1:
        i -= 1
        prod *= cute.size(t, mode=[i])
    t = cute.group_modes(t, i, r)
    return cute.group_modes(t, 1, i)


class GemmSm120(GemmSm90):
    """SM120-style GEMM using warp-level MMA instead of WGMMA.

    Key differences from SM90:
    - Uses warp-level MMA (MmaF16BF16Op m16n8k16, or MmaFP8Op m16n8k32 for
      8-bit operands) instead of WGMMA (warp-group, 128 threads)
    - Requires explicit SMEM→RMEM copy via ldmatrix before MMA
    - Thread config: num_mma_warps regular warps + 1 DMA warp
    - Pingpong: 2 warp groups of (2,2,1), each processing alternating tiles
    - fp8 (e4m3/e5m2): k-major A and B only (ldmatrix has no 8-bit
      transpose that matches the fp8 fragment). No slow-accum path: unlike
      Hopper's ~fp13 QGMMA accumulator, SM120's fp8 mma.sync f32 accumulate
      keeps ~21-22 mantissa bits (measured on RTX 5090: +1 onto 2^n survives
      through n=20, one bit short of the bf16 datapath; truncating add), so
      the per-k-tile promotion buys nothing.

    A-operand transforms (quack/operand_transform/) are supported through the
    same ``copy_block(stage_idx, b, k_tile)`` produce seam as GemmSm90's RS
    mainloop — A is always register-sourced here, so value fns / dropout wrap
    the canonical ldmatrix load, and layout-owning W4 decodes replace it.
    W4A8 fast-accum (int4smf) rides the fp8 warp MMA — the block-scaled 2x
    instruction when the tile qualifies; W4A8 promote (int4sm) stays
    SM90-only until this mainloop grows the per-k-tile promote seam.
    """

    arch = 120

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        a_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
        gather_A: bool = False,
        concat_layout: tuple | None = None,
        use_pdl: bool = True,
        split_k: int = 1,
        split_k_mode: int = SplitKMode.SERIAL,
        transform_a: Optional[Callable] = None,
    ):
        # Don't call super().__init__ — we set up our own config
        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        self.use_clc_persistence = False
        self.use_pdl = use_pdl
        self.fp8_slow_accum = False
        # The warp-MMA mainloop always consumes A from registers (ldmatrix
        # s2r), so there is no SS/RS mode split; mma_is_rs stays False for the
        # inherited __call__/_setup_attributes checks. A-operand transforms
        # (quack/operand_transform/) plug into the same copy_block seam as
        # SM90's RS mainloop — instantiated below after the register budgets.
        self.mma_is_rs = False
        self._transform_a_factory = transform_a
        self.mma_a_dtype = a_dtype
        self.gather_A = gather_A
        self.concat_layout = concat_layout or ()
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"
        if gather_A:
            assert cluster_shape_mnk[1] == 1
        self._init_split_k(split_k, split_k_mode)

        self.cluster_shape_mnk = cluster_shape_mnk
        assert len(tile_shape_mnk) in [2, 3], "CTA tile shape must be (M, N) or (M, N, K)"
        # K dimension: if user provides 3 values, use their K; otherwise default in _setup_tiled_mma.
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[:2]

        # Pingpong: 2 warp groups each with (2,2,1) atom layout
        # Non-pingpong: 1 group of 8 warps with (4,2,1) atom layout.
        # Layout-owning transforms (W4 decodes) get atom_n = 1 instead: with
        # atom_n = 2 the A fragment — and therefore the whole in-register
        # dequant — is duplicated across the N warp pair, and the 32-wide N
        # span forces tile_N >= 32 (2x padded-B traffic at decode shapes).
        # atom_m = 8 when tile_M has whole 128-row steps (8 warps, prefill),
        # else 4 (one 4-warp MMA group, 256-thread CTA, decode tiles).
        self.mma_inst_mnk = (16, 8, 16) if self.mma_a_dtype.width == 16 else (16, 8, 32)
        w4_owned = transform_a is not None and getattr(transform_a, "owned_fmt", None) is not None
        if self.pingpong:
            self.atom_layout_mnk = (2, 2, 1)
        elif w4_owned:
            self.atom_layout_mnk = (8, 1, 1) if tile_M % 128 == 0 else (4, 1, 1)
        else:
            self.atom_layout_mnk = (4, 2, 1)
        if tile_N % (16 * self.atom_layout_mnk[1]) != 0:
            # the N permutation gives each warp 16 consecutive columns: the
            # tiled MMA spans 16 * atom_n N
            raise ValueError(
                f"SM120 CTA tile N must be divisible by {16 * self.atom_layout_mnk[1]}"
            )
        # num_mma_warps = total warps doing MMA (both warp groups in pingpong)
        self.num_mma_warps = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        # For compatibility with SM90 code that uses warp groups
        self.num_threads_per_warp_group = 128
        assert self.num_mma_warps % 4 == 0
        self.mma_warp_groups = self.num_mma_warps // 4
        if self.pingpong:
            assert self.mma_warp_groups == 2
        # threads_per_cta must be a multiple of 128 (warp group size) so that
        # the DMA warp's setmaxnreg.dec.sync has a complete warp group to sync with.
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group

        self.num_mcast_ctas_a = cluster_shape_mnk[1]
        if gather_A:
            assert self.num_mcast_ctas_a == 1
        self.num_mcast_ctas_b = cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")

        # In pingpong, only 1 warp group (4 warps) participates in epilogue at a time
        self.num_epi_warps = (self.mma_warp_groups if not self.pingpong else 1) * 4
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierGemm.Epilogue),
            num_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )
        self.num_ab_load_warps = 1 if not self.gather_A else 4
        self.ab_load_warp_id = self.num_mma_warps

        if not self.gather_A:
            self.num_regs_load = 40
            self.num_regs_mma = 232
        else:
            self.num_regs_load = 56
            self.num_regs_mma = 224

        # TransformA: created after the default register budgets above so it
        # can override them (and occupancy) per its config. The transform may
        # install an aux A-side operand (per-stage strip riding the AB
        # pipeline) — same contract as GemmSm90.
        self.transform_a = None
        self.aux_a = None
        if transform_a is not None:
            self.transform_a = transform_a(self)
            self.aux_a = self.transform_a.aux

        self.ab_stage = None
        self.epi_stage = None
        self.epi_m_major = True
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None
        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def epi_smem_warp_shape_mnk(self):
        return self.atom_layout_mnk

    def _setup_tiled_mma(self):
        """Set up warp-level MMA (MmaF16BF16Op / MmaMXF8Op / MmaFP8Op) and
        tile K.

        fp8 rides the BLOCK-SCALED mma with constant unit (2^0) scale
        fragments whenever it can: on SM120 silicon the plain kind::f8f6f4
        instruction runs at HALF the rate of kind::mxf8f6f4 (measured RTX
        5090: 507 vs 1005 TFLOPS dense e4m3, with the identical ~21-22-bit
        truncating f32 accumulator — probed bit-for-bit, same +1-onto-2^n
        keep/lost boundary at n=20/21 and the same RZ signature). The SF
        operand costs one constant byte fragment per (m-atom, k-atom) and no
        loads. Constraints: same-dtype A/B (mixed e4m3 x e5m2 has no
        block-scaled form), f32 accumulator, tile_M % 128 == 0 (the SF
        fragment partition helpers assume whole 128-row SF blocks), and a
        sm_120/121 COMPILE TARGET — kind::mxf8f6f4 block_scale is the one
        SM120 fp8 instruction with no Hopper equivalent (MmaMXF8Op admits
        only sm_120a/f, sm_121a/f), so the H100 CI proxy legs
        (QUACK_ARCH=120 compiled for sm_90a) take the MmaFP8Op fallback,
        which is sm_89+ and numerically stricter there (full fp32 RNE
        accumulate vs SM120's ~21-22-bit RZ); anything else falls back to
        MmaFP8Op."""
        # mma_a_dtype, not a_dtype: a layout-owning transform's mA is a
        # storage blob (e.g. uint8) decoded to the MMA compute dtype
        tile_k_resolved = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        mma_arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        self.use_mxf8_mma = (
            self.mma_a_dtype.width == 8
            and self.mma_a_dtype == self.b_dtype
            and self.acc_dtype == Float32
            and self.cta_tile_shape_mnk[0] % 128 == 0
            # the SF layout is 4-SF (128-k at vec 32) granular
            and tile_k_resolved % 128 == 0
            # ptxas-target gate, not a dispatch gate (see docstring): mirror
            # the op's own __post_init__ admissibility check
            and mma_arch in warp.MmaMXF8Op.admissible_archs
        )
        if const_expr(self.mma_a_dtype.width == 16):
            op = warp.MmaF16BF16Op(self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk)
        elif const_expr(self.use_mxf8_mma):
            op = warp.MmaMXF8Op(self.mma_a_dtype, self.acc_dtype, cutlass.Float8E8M0FNU)
        else:
            op = warp.MmaFP8Op(self.mma_a_dtype, self.acc_dtype, self.mma_inst_mnk)
        tC = cute.make_layout(self.atom_layout_mnk)
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        # We want each warp to have 16 consecutive elements in the N direction, for STSM
        # and for gated epilogue.
        permutation_n = cute.make_ordered_layout((self.mma_inst_mnk[1], atom_n, 2), order=(0, 2, 1))
        permutation_mnk = (
            atom_m * self.mma_inst_mnk[0],
            permutation_n,
            atom_k * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)
        tile_k = (
            self.cta_tile_shape_mnk[2]
            if self.cta_tile_shape_mnk[2] > 0
            else self.mma_inst_mnk[2] * 4
        )
        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % self.mma_inst_mnk[2] == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({self.mma_inst_mnk[2]})"
        )
        if self.transform_a is not None and self.transform_a.tile_k is not None:
            assert tile_k == self.transform_a.tile_k, (
                f"transform_a requires tile_K == {self.transform_a.tile_k}, got {tile_k}"
            )
        self.cta_tile_shape_mnk = (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], tile_k)

    def canonical_a_load(self, tiled_mma, sA, tidx, tCrA):
        """The canonical A produce for the warp-MMA mainloop: the same
        ldmatrix seam as SM90 RS (identical fragment atoms), with the smem
        major passed explicitly — the warp MMA ops carry no major mode
        (operand layout is fixed K-major; the smem major only picks LDSM vs
        LDSM.T). fp8 A rides the same 16-bit LDSM atom typed at the fp8
        element (a k-major byte pair is one 16-bit unit; the m16n8k32 fp8
        fragment is the m16n8k16 16-bit fragment at twice the k density), so
        it is k-major only — enforced by the inherited __call__ checks."""
        atom = None
        if const_expr(self.mma_a_dtype.width == 8):
            atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.mma_a_dtype
            )
        return quack_sm90_utils.canonical_a_load_s2r(
            tiled_mma,
            sA,
            tidx,
            tCrA,
            position_independent=True,
            transpose=self.a_layout.is_m_major_a(),
            atom=atom,
        )

    # __call__, _setup_attributes, make_ab_pipeline, make_epi_store_pipeline,
    # make_sched_pipeline, epilogue are all inherited from GemmSm90.

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: Optional[cute.CopyAtom],
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: Optional[cute.CopyAtom],
        mD_mnl: Optional[cute.Tensor],
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: Optional[cute.Tensor],
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        # plain Layout for layout-owning transforms (unswizzled blob smem)
        a_smem_layout: Union[cute.ComposedLayout, cute.Layout],
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_c_smem_layout: cute.ComposedLayout,
        # aux A-side operand slots (e.g. a transform's scale-factor strip
        # riding the AB pipeline, or a raw dropout seed tensor)
        tma_atom_aux_a: Optional[cute.CopyAtom],
        mAuxA_mkl: Optional[cute.Tensor],
        aux_a_smem_layout: Optional[cute.Layout],
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
    ):
        from cutlass.cute.experimental import iket

        varlen_m = const_expr(varlen_params.cu_seqlens_m is not None)
        varlen_k = const_expr(varlen_params.cu_seqlens_k is not None)
        if const_expr(self.gather_A):
            assert varlen_m or varlen_k
        has_D = const_expr(mD_mnl is not None)
        has_C = const_expr(mC_mnl is not None)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors
        if warp_idx == self.ab_load_warp_id:
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d, tma_atom_c, tma_atom_aux_a):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
        )
        epi_pipeline = None
        has_epi_load = const_expr(self.epi_c_stage > 0)
        if const_expr(has_epi_load):
            epi_pipeline = self.make_epi_pipeline(tx_count=self.epi_load_bytes_per_stage)
        sched_pipeline = None
        sched_data = None
        if const_expr(self.is_persistent):
            sched_pipeline = self.make_sched_pipeline(
                cluster_layout_mnk,
                # split_k > 1 makes per-tile k-tile counts dynamic, so pingpong consumes
                # work tiles one at a time, exactly like varlen_k.
                varlen_k=varlen_k or self.split_k > 1,
            )
            # Keep scheduler scratch out of SharedStorage. A small buffer before
            # the 1024-byte aligned epilogue tensors can add a 1 KiB pad; CLC
            # responses also use i128 copies, so this stays 16-byte aligned.
            # No drain-mailbox tail (+6 Int32, cf. gemm_sm100): this kernel never
            # calls cancel_pending_tail — add the tail if that ever changes.
            sched_data = smem.allocate_tensor(
                Int32,
                cute.make_layout((4, self.sched_stage)),
                byte_alignment=16,
                partition=SmemPartition.RESERVED,
            )

        # Cluster sync
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        # SMEM tensors
        a_owned = const_expr(self.transform_a is not None and self.transform_a.owns_a_layout)
        if const_expr(not a_owned):
            sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        else:
            # TMA-facing staged blob view (plain layout, no swizzle); the
            # transform's per-thread math view recasts the same bytes inside
            # make_copy_block
            sA = storage.sA.get_tensor(a_smem_layout)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sAuxA = None
        if const_expr(self.aux_a is not None):
            sAuxA = storage.sAuxA.get_tensor(aux_a_smem_layout)
        sD = None
        if const_expr(has_D):
            sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        sC = None
        if const_expr(has_C):
            sC = storage.sC.get_tensor(epi_c_smem_layout.outer, swizzle=epi_c_smem_layout.inner)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            # Only used if not varlen_m; a layout-owning transform's mA is a
            # storage blob, so kernel-M comes from D instead.
            len_m_static=Int32(
                (
                    cute.size(mA_mkl, mode=[0])
                    if const_expr(not a_owned)
                    else cute.size(mD_mnl, mode=[0])
                )
                if varlen_k or varlen_params.mAIdx is None
                else varlen_params.mAIdx.shape[0]
            ),
            len_k_static=Int32(cute.size(mB_nkl, mode=[1])),
            len_n_static=Int32(cute.size(mB_nkl, mode=[0])),
        )

        TileSchedulerCls = partial(
            TileSchedulerCls.create, tile_sched_params, sched_data, sched_pipeline
        )

        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                # block_copy's lowering wants the coordinate held fixed by the
                # multicast mask: A is same-M across N peers, while B is
                # same-N across M peers. Degenerate cluster dimensions are
                # left for the compiler lowering to simplify.
                a_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "M",
                }
                b_tma_multicast = {
                    "cluster_shape": self.cluster_shape_mnk[:2],
                    "multicast_dim": "N",
                }

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ag_last_gate = Int32(-1)  # 1-entry satisfied-gate cache (see ag_wait_m_tile)
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                    # AllGather+GEMM: block until this tile's M-shard of A has
                    # been pushed into local HBM by the owner rank (see
                    # gemm_sm90.py — same shared-code gate).
                    if const_expr(getattr(tile_sched_params, "ag", None) is not None):
                        iket.range_push("ag_wait")
                        ag_last_gate = ag_wait_m_tile(
                            tile_sched_params,
                            tile_coord_mnkl[0],
                            self.cluster_shape_mnk[0],
                            ag_last_gate,
                        )
                        iket.range_pop()
                    iket.range_push("tma_load")
                    # Local_tile partition global tensors
                    copy_A, prefetch_A = None, None
                    if const_expr(a_owned):
                        # the transform owns A's gmem interpretation
                        gA_owned = self.transform_a.a_gmem_slice(mA_mkl, tile_coord_mnkl, batch_idx)
                        copy_A = copy_utils.tma_get_block_copy_fn(
                            tma_atom_a,
                            src_tensor=gA_owned,
                            dst_tensor=sA,
                            tma_multicast=a_tma_multicast,
                        )
                    elif const_expr(not self.gather_A):
                        mA_mk = varlen_manager.offset_batch_A(mA_mkl, batch_idx)
                        # (bM, bK, RestK)
                        gA_mk = cute.local_tile(
                            mA_mk,
                            cute.select(self.cta_tile_shape_mnk, [0, 2]),
                            (tile_coord_mnkl[0], None),
                        )
                        #  TMA load A partition_S/D
                        copy_A = copy_utils.tma_get_block_copy_fn(
                            tma_atom_a,
                            src_tensor=gA_mk,
                            dst_tensor=sA,
                            tma_multicast=a_tma_multicast,
                        )
                    else:
                        copy_A, prefetch_A = self._make_gather_A_copy(
                            mA_mkl, sA, varlen_manager, tile_coord_mnkl, batch_idx
                        )
                    copy_AuxA = None
                    if const_expr(self.aux_a is not None):
                        # aux A-side operand: one box per k-tile alongside A/B
                        gAux = self.aux_a.gmem_slice(mAuxA_mkl, tile_coord_mnkl, batch_idx)
                        copy_AuxA = copy_utils.tma_get_block_copy_fn(
                            tma_atom_aux_a,
                            src_tensor=gAux,
                            dst_tensor=sAuxA,
                            # small-box aux operands (e.g. 128 B scale strips)
                            # may opt out of the A-side multicast: each CTA
                            # loads its own copy instead of splitting the box
                            tma_multicast=a_tma_multicast
                            if const_expr(getattr(self.aux_a, "multicast", True))
                            else None,
                        )
                    # (bN, bK, RestK)
                    gB_nk = cute.local_tile(
                        varlen_manager.offset_batch_B(mB_nkl, batch_idx),
                        cute.select(self.cta_tile_shape_mnk, [1, 2]),
                        (tile_coord_mnkl[1], None),
                    )
                    # TMA load B partition_S/D
                    copy_B = copy_utils.tma_get_block_copy_fn(
                        tma_atom_b,
                        src_tensor=gB_nk,
                        dst_tensor=sB,
                        tma_multicast=b_tma_multicast,
                    )
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    k_tile_start, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                        k_tile_total, split_idx
                    )
                    if const_expr(not self.gather_A):
                        ab_producer_state = self.load_tma(
                            ab_pipeline,
                            ab_producer_state,
                            [copy_A, copy_B, copy_AuxA],
                            k_tile_cnt,
                            k_tile_start=k_tile_start,
                        )
                    else:
                        ab_producer_state = self.load_AB_gather_A(
                            ab_pipeline,
                            ab_producer_state,
                            copy_A,
                            prefetch_A,
                            copy_B,
                            k_tile_cnt,
                            varlen_m=varlen_m,
                        )
                    iket.range_pop()
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong and not varlen_k and self.split_k == 1):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        # =====================================================================
        # MMA warps
        # =====================================================================
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            # For pingpong, adjust tidx to within-warp-group index
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group

            # ldmatrix copy atom for SMEM → RMEM (B side; A goes through the
            # copy_block seam below)
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)

            # Make fragments
            thr_mma = tiled_mma.get_slice(tidx)
            if const_expr(not a_owned):
                acc, tCsA, tCsB, tCrA, tCrB = sm80_utils.partition_fragment_ABC(
                    thr_mma, self.cta_tile_shape_mnk, sA, sB
                )
            else:
                # mA is a storage blob: the A fragment can't be partitioned
                # from sA — build it from the tile shape (the transform's
                # copy_block fills it in fragment order)
                acc = cute.make_rmem_tensor(
                    thr_mma.partition_shape_C(self.cta_tile_shape_mnk[:2]), Float32
                )
                if const_expr(not self.use_mxf8_mma):
                    tCrA = thr_mma.make_fragment_A(
                        thr_mma.partition_shape_A(cute.select(self.cta_tile_shape_mnk, [0, 2]))
                    )
                else:
                    # the block-scaled atom's fragment verifier rejects
                    # shape-built fragments; partition a dummy k-major
                    # (tile_M, tile_K) view instead (only shapes are used —
                    # the pointer is never dereferenced) and fragment that.
                    cA_fake = cute.make_tensor(
                        cute.recast_ptr(sB.iterator, dtype=self.mma_a_dtype),
                        cute.make_ordered_layout(
                            cute.select(self.cta_tile_shape_mnk, [0, 2]), order=(1, 0)
                        ),
                    )
                    tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(cA_fake))
                tCsB = thr_mma.partition_B(sB)
                tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])

            # Block-scaled fp8 (MmaMXF8Op): constant unit-scale fragments.
            # The partition helpers only consume layout shapes, so a dummy
            # tensor over any pointer serves; the fragments are filled with
            # ue8m0 1.0 (0x7F) once and never reloaded — the block-scaled
            # instruction is purely a 2x-rate fp8 mma here.
            tCrSFA, tCrSFB = None, None
            if const_expr(self.use_mxf8_mma):
                sfa_layout = blockscaled_layout.sm120_make_smem_layout_sfa(
                    tiled_mma, self.cta_tile_shape_mnk, 32, 1
                )
                # the SF blob is 128-N granular (SFB layouts assert it); for
                # tile_N < 128 CUTLASS bumps the SFB tile and broadcast-
                # slices — with unit scales any N-slice is valid, so bump
                # here and restrict the fragment's N mode to MMA_N below
                sfb_tile = (
                    self.cta_tile_shape_mnk[0],
                    max(self.cta_tile_shape_mnk[1], 128),
                    self.cta_tile_shape_mnk[2],
                )
                sfb_layout = blockscaled_layout.sm120_make_smem_layout_sfb(
                    tiled_mma, sfb_tile, 32, 1
                )
                sf_ptr = cute.recast_ptr(sB.iterator, dtype=cutlass.Float8E8M0FNU)
                sSFA_like = cute.make_tensor(sf_ptr, sfa_layout)
                sSFB_like = cute.make_tensor(sf_ptr, sfb_layout)
                tCrSFA = blackwell_helpers.partition_fragment_SFA(
                    sSFA_like[None, None, 0], thr_mma, tidx
                )
                tCrSFB = blackwell_helpers.partition_fragment_SFB(
                    sSFB_like[None, None, 0], thr_mma, tidx
                )

                # Normalize to (V, MN, K): the raw fragments grow extra
                # rest-M/N modes past 128 rows (e.g. tile_N=256 -> rank 4
                # with a size-2 block mode BEFORE K) — grouping blindly from
                # mode 2 folds that block mode into K and scrambles the
                # per-k-block slices (read-out-of-fragment ue8m0 bytes decode
                # to NaN scales). Walk K off the right by size instead.
                tCrSFA = _sf_group_vmk(tCrSFA, self.cta_tile_shape_mnk[2] // 32)
                tCrSFB = _sf_group_vmk(tCrSFB, self.cta_tile_shape_mnk[2] // 32)
                if const_expr(cute.size(tCrSFB, mode=[1]) != cute.size(tCrB, mode=[1])):
                    # tile_N < 128: the bumped SFB fragment has more N atoms
                    # than the tile; restrict to MMA_N (any slice is valid —
                    # every SF byte is the same unit scale)
                    tCrSFB = cute.composition(
                        tCrSFB,
                        (None, cute.make_layout(cute.size(tCrB, mode=[1])), None),
                    )
                cute.recast_tensor(tCrSFA, cutlass.Int8).fill(127)
                cute.recast_tensor(tCrSFB, cutlass.Int8).fill(127)

            # A produce seam: the canonical ldmatrix s2r load, or a
            # transform's own produce (e.g. blob LDS + dequant) — same
            # copy_block(stage_idx, b, k_tile) contract as GemmSm90's RS
            # mainloop.
            if const_expr(self.transform_a is not None):
                copy_block = self.transform_a.make_copy_block(
                    tiled_mma,
                    sA,
                    tCrA,
                    tidx,
                    warp_group_idx,
                    sAux=sAuxA,
                    mAux=mAuxA_mkl if const_expr(self.transform_a.aux_raw) else None,
                )
            else:
                copy_block = self.canonical_a_load(tiled_mma, sA, tidx, tCrA)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            k_tile_cnt_static = cute.ceil_div(
                cute.size(mA_mkl, mode=[1]), self.cta_tile_shape_mnk[2]
            )
            c_tile_cnt = cute.size(cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile))

            ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            epi_store_pipeline = self.make_epi_store_pipeline()
            epi_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_c_stage
            )
            epi_producer_state = make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_c_stage
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    if const_expr(not varlen_k and self.split_k == 1):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                    else:
                        # varlen_k and split_k > 1 both make the per-tile k-tile count dynamic
                        batch_idx_pp, split_idx_pp = (
                            work_tile.tile_idx[3],
                            work_tile.tile_idx[2],
                        )
                        len_k = varlen_manager.len_k(batch_idx=batch_idx_pp)
                        k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                            k_tile_total, split_idx_pp
                        )
                        ab_read_state.advance_iters(k_tile_cnt)
                        # Under split-K, only finalizer tiles run the epilogue (and thus
                        # produce/consume C stages); the peer advance must match.
                        c_cnt = Int32(c_tile_cnt)
                        if const_expr(
                            self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE
                        ):
                            if split_idx_pp != self.split_k - 1:
                                c_cnt = Int32(0)
                        epi_read_state.advance_iters(c_cnt)
                        epi_producer_state.advance_iters(c_cnt)
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
            while work_tile.is_valid_tile:
                # (pid_m, pid_n, split_idx | None, batch_idx), decoded by the scheduler
                tile_coord_mnkl = work_tile.tile_idx
                batch_idx, split_idx = tile_coord_mnkl[3], tile_coord_mnkl[2]
                len_k = varlen_manager.len_k(batch_idx)
                k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                k_tile_start_mma, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                    k_tile_total, split_idx
                )
                if const_expr(self.transform_a is not None):
                    if const_expr(self.transform_a.uses_work_tile):
                        # per-work-tile register state (e.g. dropout's per-row
                        # RNG coordinates); every copy_block until the next
                        # hook — incl. the slot-0 preloads — is this tile's
                        self.transform_a.on_work_tile(tile_coord_mnkl)
                acc.fill(0.0)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                iket.range_push("mma")
                ab_read_state = self.mma(
                    ab_pipeline,
                    ab_read_state,
                    tiled_mma,
                    acc,
                    k_tile_cnt,
                    copy_block,
                    smem_tiled_copy_B,
                    tCsB_copy_view,
                    tCrA,
                    tCrB,
                    k_tile_start=k_tile_start_mma,
                    tCrSFA=tCrSFA,
                    tCrSFB=tCrSFB,
                )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                iket.range_pop()

                # ============================================================
                # EPILOGUE — reuse SM90's epilogue flow
                # ============================================================
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                iket.range_push("epilogue")

                copy_D = None
                if const_expr(has_D):
                    # Staged split-K: D is the f32 partials workspace, whose batch mode is the
                    # combined (l * split_k + split) index from the scheduler.
                    d_batch_idx = batch_idx
                    if const_expr(self.split_k > 1 and self.split_k_mode == SplitKMode.SEPARATE):
                        d_batch_idx = tile_scheduler.get_combined_batch_idx(batch_idx, split_idx)
                    copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_d,
                        varlen_manager.offset_batch_epi(mD_mnl, d_batch_idx),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sD,
                        tile_coord_mnkl,
                    )

                copy_C = None
                if const_expr(has_C):
                    copy_C_fn, _, _ = self.epilog_gmem_copy_and_partition(
                        tma_atom_c,
                        varlen_manager.offset_batch_epi(mC_mnl, tile_coord_mnkl[3]),
                        self.cta_tile_shape_mnk[:2],
                        self.epi_tile,
                        sC,
                        tile_coord_mnkl,
                    )
                    copy_C = copy_utils.tma_producer_copy_fn(copy_C_fn, epi_pipeline)
                if const_expr(has_epi_load):
                    tile_load_copy_fns = self.epi_tile_load_g2s_copy_fns(
                        epilogue_params,
                        epi_smem_tensors,
                        tile_coord_mnkl,
                        varlen_manager,
                        epi_pipeline,
                    )
                    copy_C = copy_utils.chain_tma_producer_copy_fns((copy_C, *tile_load_copy_fns))

                d_dtype_for_layout = self.d_dtype if self.d_dtype is not None else cutlass.BFloat16
                tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                    tiled_mma, self.d_layout, d_dtype_for_layout, sD, tidx
                )
                # (R2S, R2S_M, R2S_N, (epi_M, epi_N))
                tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
                load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                if const_expr(has_C):
                    tiled_copy_s2r, tRS_rC, tSR_rC, tSR_sC = self.epilog_smem_load_and_partition(
                        tiled_mma, self.c_layout, self.c_dtype, sC, tRS_rD.layout, tidx
                    )
                else:
                    tiled_copy_s2r, tSR_sC, tRS_rC, tSR_rC = None, None, None, None

                self.epi_visit_acc(epilogue_params, acc, tiled_mma, tile_coord_mnkl, tidx)

                # Split-K (serial/parallel): non-finalizing splits commit raw f32 partials
                # to the tile's workspace and skip the epilogue; the last split waits for
                # the tile's completion flag and runs the full epilogue on the summed
                # accumulator (CUTLASS-3.x stream-K fixup semantics).
                epi_fn = partial(
                    self.epilogue,
                    epilogue_params,
                    epi_smem_tensors,
                    epi_pipeline,
                    epi_store_pipeline,
                    epi_read_state,
                    epi_producer_state,
                    self.epi_tile,
                    # load_acc_subtile is the one argument left unbound
                    tRS_rD=tRS_rD,
                    tRS_rC=tRS_rC,
                    tiled_copy_t2r=None,  # Sm100 only
                    tiled_copy_r2s=tiled_copy_r2s,
                    tRS_sD=tRS_sD,
                    tiled_copy_s2r=tiled_copy_s2r,
                    tSR_rC=tSR_rC,
                    tSR_sC=tSR_sC,
                    copy_D=copy_D,
                    copy_C=copy_C,
                    tile_coord_mnkl=tile_coord_mnkl,
                    varlen_manager=varlen_manager,
                    epilogue_barrier=self.epilogue_barrier,
                    tile_scheduler=tile_scheduler,
                    tidx=tidx,
                    is_tma_warp=is_tma_warp,
                )
                epi_read_state, epi_producer_state = self.epilogue_split_k(
                    epilogue_params,
                    epi_fn,
                    load_acc_subtile,
                    tRS_rD,
                    self.epi_tile,
                    epi_read_state,
                    epi_producer_state,
                    epi_store_pipeline,
                    tile_coord_mnkl,
                    self.epilogue_barrier,
                    tidx,
                    is_tma_warp,
                )

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signaling
                    # the next WG's epilogue.
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                iket.range_pop()

                if const_expr(not self.pingpong):
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                else:  # Skip a tile for pingpong
                    # Update starting load/store/mainloop pipeline states for the next tile
                    if const_expr(not varlen_k and self.split_k == 1):
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        ab_read_state.advance_iters(k_tile_cnt_static)
                        tile_scheduler.advance_to_next_work(advance_count=self.mma_warp_groups)
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            batch_idx_pp, split_idx_pp = (
                                work_tile.tile_idx[3],
                                work_tile.tile_idx[2],
                            )
                            len_k = varlen_manager.len_k(batch_idx=batch_idx_pp)
                            k_tile_total = cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            _, k_tile_cnt = tile_scheduler.get_split_k_tile_range(
                                k_tile_total, split_idx_pp
                            )
                            ab_read_state.advance_iters(k_tile_cnt)
                            # Under split-K, only finalizer tiles run the epilogue (and
                            # thus produce/consume C stages); the peer advance must match.
                            c_cnt = Int32(c_tile_cnt)
                            if const_expr(
                                self.split_k > 1 and self.split_k_mode != SplitKMode.SEPARATE
                            ):
                                if split_idx_pp != self.split_k - 1:
                                    c_cnt = Int32(0)
                            epi_read_state.advance_iters(c_cnt)
                            epi_producer_state.advance_iters(c_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

            # Wait for D store complete
            if const_expr(not self.pingpong):
                if is_tma_warp:
                    epi_store_pipeline.producer_tail()

    @cute.jit
    def mma(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        k_tile_cnt: Int32,
        copy_block: Callable,
        smem_tiled_copy_B: cute.TiledCopy,
        tCsB_copy_view: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        k_tile_start: Int32 = 0,
        tCrSFA: Optional[cute.Tensor] = None,
        tCrSFB: Optional[cute.Tensor] = None,
    ) -> cutlass.pipeline.PipelineState:
        """Warp-level MMA mainloop: A produced per k16 block through the
        ``copy_block(stage_idx, b, k_tile)`` seam (canonical ldmatrix s2r, or
        a transform's decode; ``k_tile`` is the GLOBAL k-tile index of the
        produced block, split-k correct via ``k_tile_start``), B via
        ldmatrix, then warp MMA. Same produce rhythm as CUTLASS's SM120
        collective and GemmSm90.mma_rs_interleaved: produce block k+1 (slot 0
        of the next stage at the tile's last block), then MMA block k — the
        warp-synchronous mma.sync needs none of the WGMMA commit-group/wait
        discipline, so the seam contract is the schedule alone."""
        tCrB_copy_view = smem_tiled_copy_B.retile(tCrB)
        load_sB = partial(cute.copy, smem_tiled_copy_B)

        num_k_blocks = cute.size(tCrA, mode=[2])
        kt = Int32(k_tile_start)  # global k-tile index of the tile being consumed
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)

        # Load first k-block
        stage = ab_read_state.index
        tCsB_p = tCsB_copy_view[None, None, None, stage]
        copy_block(stage, 0, kt)
        load_sB(tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

        for k_tile in cutlass.range(k_tile_cnt - 1, unroll=1):
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
                    stage = ab_read_state.index
                    tCsB_p = tCsB_copy_view[None, None, None, stage]
                    ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                # the wrap load is the NEXT tile's slot-0 preload
                copy_block(stage, k_next, kt + 1 if k == num_k_blocks - 1 else kt)
                load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                if const_expr(tCrSFA is not None):
                    # block-scaled mma: the constant unit-scale SF fragments
                    # ride as list operands (see kernel())
                    cute.gemm(
                        tiled_mma,
                        acc,
                        [tCrA[None, None, k], tCrSFA[None, None, k]],
                        [tCrB[None, None, k], tCrSFB[None, None, k]],
                        acc,
                    )
                else:
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            kt += 1

        # Last k-tile (hoisted)
        if 0 < k_tile_cnt:
            for k in cutlass.range_constexpr(num_k_blocks):
                k_next = 0 if k + 1 == num_k_blocks else k + 1
                if const_expr(k == num_k_blocks - 1):
                    # TMA writes this smem stage through the async proxy, while ldmatrix
                    # reads it through the generic proxy. Fence before release so the
                    # producer's next async-proxy write cannot race those reads; sync the
                    # warp because only one lane signals the empty mbarrier.
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    ab_pipeline.consumer_release(ab_read_state)
                    ab_read_state.advance()
                if const_expr(k_next > 0):
                    copy_block(stage, k_next, kt)
                    load_sB(tCsB_p[None, None, k_next], tCrB_copy_view[None, None, k_next])
                if const_expr(tCrSFA is not None):
                    cute.gemm(
                        tiled_mma,
                        acc,
                        [tCrA[None, None, k], tCrSFA[None, None, k]],
                        [tCrB[None, None, k], tCrSFB[None, None, k]],
                        acc,
                    )
                else:
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)

        return ab_read_state

    @staticmethod
    def _compute_tile_shape_or_override(
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param cta_tile_shape_mnk: CTA tile shape (M,N,K)
        :type cta_tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        n_perf = 64 if element_type is not None and element_type.width == 8 else 32
        # The epilogue tile must cover the tiled MMA's M span (atom_m * 16
        # rows): a subtile smaller than the warp footprint makes the r2s
        # partition wrap across warps (row-permuted/duplicated corruption —
        # hit by the (8,1,1) W4 layout whose span is 128 > the default 64).
        m_span = atom_layout_mnk[0] * 16
        tile_m = max(math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0])), m_span)
        tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)
