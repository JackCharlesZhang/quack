# Copyright (c) 2025, Wentao Guo, Tri Dao.
from typing import Tuple, Optional, Callable
from functools import partial
from dataclasses import dataclass

from torch import Tensor

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32, Float32, Boolean, const_expr
import cutlass.torch as cutlass_torch

from quack.cute_dsl_utils import ArgumentsBase, ParamsBase
from quack.varlen_utils import VarlenManager
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
from quack.gemm_wrapper_utils import GemmWrapperBase
import quack.sm90_utils as sm90_utils
import quack.copy_utils as copy_utils
import quack.activation


class GemmActMixin:
    num_epi_tensormaps: int = 1

    @dataclass
    class EpilogueArguments(ArgumentsBase):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32] = None
        beta: Optional[Float32] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        tma_atom_postact: cute.CopyAtom
        mPostAct_mnl: cute.Tensor
        epi_postact_smem_layout_staged: cute.ComposedLayout
        epi_tile_postact: cute.Tile
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32] = None
        beta: Optional[Float32] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)

        self.cta_tile_shape_postact_mn = self.cta_tile_shape_mnk[:2]
        epi_tile_postact = self.epi_tile
        utils_cls = sm100_utils if self.arch == 100 else sm90_utils
        epi_postact_smem_layout_staged = utils_cls.make_smem_layout_epi(
            self.postact_dtype, self.postact_layout, epi_tile_postact, self.epi_stage
        )
        tma_atom_postact, tma_tensor_postact = self._make_tma_epi_atoms_and_tensors(
            args.mPostAct,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            op_type="store",
        )
        return self.EpilogueParams(
            tma_atom_postact,
            tma_tensor_postact,
            epi_postact_smem_layout_staged,
            epi_tile_postact,
            args.act_fn,
            args.alpha,
            args.beta,
        )

    def epi_get_tma_atoms(
        self, params: EpilogueParams, *, loc=None, ip=None
    ) -> list[cute.CopyAtom]:
        return [params.tma_atom_postact]

    def epi_get_tensormap_update_shapes_orders(
        self,
        params: EpilogueParams,
        cu_seqlens_m: Optional[cute.Tensor],
        batch_idx: Int32,
        *,
        loc=None,
        ip=None,
    ) -> tuple[list[Int32], list[int]]:
        shapes = [cu_seqlens_m[batch_idx + 1] if cu_seqlens_m is not None else None]
        orders = [0 if const_expr(self.postact_layout.is_m_major_c()) else 1]
        return shapes, orders

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: EpilogueArguments,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        postact_dtype = args.mPostAct.element_type
        postact_bytes_per_stage = cute.size(cute.shape(epi_tile)) * (postact_dtype.width // 8)
        return postact_bytes_per_stage

    def epi_get_smem_struct(self, params: EpilogueParams):
        @cute.struct
        class EpiSharedStorage:
            sPostAct: cute.struct.Align[
                cute.struct.MemRange[
                    self.postact_dtype, cute.cosize(params.epi_postact_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        return EpiSharedStorage

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        sPostAct = storage.epi.sPostAct.get_tensor(
            params.epi_postact_smem_layout_staged.outer,
            swizzle=params.epi_postact_smem_layout_staged.inner,
        )
        return (sPostAct,)

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        tma_desc_epi_ptrs: list[Optional[cute.Pointer]],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_store_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: cutlass.pipeline.PipelineState,
        epi_tile: cute.Tile,
        load_acc_subtile: Callable,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_t2r: Optional[cute.TiledCopy],  # Only for Sm100
        tiled_copy_r2s: cute.TiledCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.TiledCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        copy_C: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)

        tma_atom_postact = params.tma_atom_postact
        mPostAct_mnl = params.mPostAct_mnl
        (sPostAct,) = epi_smem_tensors
        get_smem_store_op = (
            partial(sm100_utils.get_smem_store_op, tiled_tmem_load=tiled_copy_t2r)
            if self.arch == 100
            else sm90_utils_og.sm90_get_smem_store_op
        )
        copy_atom_postact_r2s = get_smem_store_op(
            self.postact_layout, self.postact_dtype, self.acc_dtype
        )
        # tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        # tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_C_atom)
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        (tma_desc_postact_ptr,) = tma_desc_epi_ptrs
        batch_idx = tile_coord_mnkl[3]
        copy_postact, _, _ = self.epilog_gmem_copy_and_partition(
            tma_atom_postact,
            varlen_manager.offset_batch_epi(mPostAct_mnl, batch_idx),
            self.cta_tile_shape_postact_mn,
            params.epi_tile_postact,
            sPostAct,
            tile_coord_mnkl,
            tma_desc_ptr=tma_desc_postact_ptr,
        )

        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
        epi_tile_num = cute.size(epi_tile_shape)
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        if const_expr(copy_C is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from acc to D registers
            load_acc_subtile(tRS_rD, epi_idx)
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(copy_C is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx + self.epi_c_stage)
                if is_tma_warp:
                    epi_pipeline.producer_acquire(epi_producer_state)
                    copy_C(src_idx=gmem_coord, producer_state=epi_producer_state)
                    epi_pipeline.producer_commit(epi_producer_state)
                epi_producer_state.advance()
            tRS_rPostAct = self.epi_visit_acc_subtile(params, tRS_rD, tRS_rC)
            epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
            # Copy from D registers to shared memory
            if const_expr(has_D):
                copy_utils.cvt_copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, epi_buffer])
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            epilogue_barrier.arrive_and_wait()
            # Get the global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(src_idx=epi_buffer, dst_idx=gmem_coord)
                copy_postact(src_idx=epi_buffer, dst_idx=gmem_coord)
                epi_store_pipeline.producer_commit()
                epi_store_pipeline.producer_acquire()
            epilogue_barrier.arrive_and_wait()

        return epi_read_state, epi_producer_state

    @cute.jit
    def epi_visit_acc_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(params.alpha is not None):
            tRS_rD.store(tRS_rD.load() * params.alpha)
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                tRS_rD.store(tRS_rD.load() + tRS_rC.load().to(tRS_rD.element_type))
            else:
                tRS_rD.store(tRS_rD.load() + params.beta * tRS_rC.load().to(tRS_rD.element_type))
        # Apply activation function if provided
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_fragment(tRS_rD.layout.shape, self.acc_dtype)
            if const_expr(self.arch < 100):
                for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                    tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
            else:
                for i in cutlass.range(cute.size(tRS_rPostAct) // 2, unroll_full=True):
                    tRS_rPostAct[2 * i], tRS_rPostAct[2 * i + 1] = params.act_fn(
                        (tRS_rD[2 * i], tRS_rD[2 * i + 1])
                    )
        else:
            tRS_rPostAct = tRS_rD
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


class GemmActSm90(GemmActMixin, GemmSm90):
    pass


class GemmActSm100(GemmActMixin, GemmSm100):
    pass


act_fn_map = {
    None: None,
    "relu": quack.activation.relu,
    "relu_sq": quack.activation.relu_sq,
    "gelu_tanh_approx": quack.activation.gelu_tanh_approx,
}


def gemm_act(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
) -> None:
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in act_fn_map, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, additional_tensors={"PostAct": PostAct}, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
    )
    GemmWrapperBase.permute_tensors(tensor_infos, varlen_m=cu_seqlens_m is not None)
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
        "PostAct": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9, 10], "Only SM90 and SM100 are supported"
    GemmCls = GemmActSm100 if device_capacity[0] > 9 else GemmActSm90

    acc_dtype = cutlass.Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors(tensor_infos, major_configs)
    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(tensor_infos["PostAct"].cute_tensor, act_fn)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore, max_swizzle_size=max_swizzle_size
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
        pingpong,
    )

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        activation,
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        max_swizzle_size,
        cu_seqlens_m is not None,
        A_idx is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_act.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


gemm_act.compile_cache = {}
