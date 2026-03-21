# Copyright (c) 2025, Tri Dao.
"""Composable epilogue operations (EpiOps) for GEMM kernels.

Each EpiOp encapsulates a single tensor kind's behavior across the epilogue lifecycle:
smem allocation, begin (one-time per-tile setup), begin_loop (per-subtile extraction),
end (cleanup).

The ops are composed via ComposableEpiMixin which iterates over a static _epi_ops tuple
to generate epi_smem_bytes_per_stage, epi_get_smem_struct, epi_get_smem_tensors,
epi_begin, and epi_begin_loop automatically.
"""

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr

from quack.epi_utils import assume_stride_divisibility, setup_epi_tensor
from quack.sm90_utils import partition_for_epilogue
import quack.utils as utils
import quack.copy_utils as copy_utils


class EpiContext:
    """Shared context passed to EpiOp.begin methods. Bundles common arguments."""

    __slots__ = (
        "epi_tile",
        "tiled_copy_t2r",
        "tiled_copy_r2s",
        "tile_coord_mnkl",
        "varlen_manager",
        "epilogue_barrier",
        "tidx",
        "partition_for_epilogue_fn",
        "num_epi_threads",
        "batch_idx",
        "tile_M",
        "tile_N",
    )

    def __init__(
        self,
        gemm,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        epilogue_barrier,
        tidx,
    ):
        self.epi_tile = epi_tile
        self.tiled_copy_t2r = tiled_copy_t2r
        self.tiled_copy_r2s = tiled_copy_r2s
        self.tile_coord_mnkl = tile_coord_mnkl
        self.varlen_manager = varlen_manager
        self.epilogue_barrier = epilogue_barrier
        self.tidx = tidx
        self.tile_M = gemm.cta_tile_shape_mnk[0]
        self.tile_N = gemm.cta_tile_shape_mnk[1]
        self.batch_idx = tile_coord_mnkl[3]
        self.num_epi_threads = gemm.num_epi_warps * cute.arch.WARP_SIZE
        self.partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )


class EpiOp:
    """Base class for composable epilogue operations."""

    def __init__(self, name):
        self.name = name

    # --- Host-side: args → params ---
    def to_params(self, gemm, args):
        """Convert this op's arg field(s) to param dict entries.
        Returns dict of {param_name: value}. Like EVT's to_underlying_arguments."""
        return {}

    # --- Host-side: smem allocation ---
    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        """Bytes of smem needed per stage. arg_tensor is the EpilogueArguments field."""
        return 0

    def smem_struct_field(self, gemm, params):
        """Return (field_name, field_type) for @cute.struct, or None if no smem needed.
        params is the full EpilogueParams object."""
        return None

    def get_smem_tensor(self, gemm, params, storage_epi):
        """Extract smem tensor from storage.epi. Returns tensor or None.
        params is the full EpilogueParams object."""
        return None

    def tma_atoms(self, gemm, params):
        """Return list of TMA atoms for this op."""
        return []

    # --- Device-side: kernel execution ---
    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        """One-time per-tile setup. Returns state for begin_loop."""
        return None

    def begin_loop(self, gemm, state, epi_coord):
        """Per-subtile extraction. Returns value for epi_visit_subtile."""
        return state

    def needs_async_fence(self):
        """Whether this op issues async copies that need a fence."""
        return False

    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Cleanup after all subtiles (reductions, direct writes)."""
        pass


class Scalar(EpiOp):
    """Loads a scalar value or device pointer once per tile. No smem."""

    def __init__(self, name, dtype=None):
        super().__init__(name)
        self.dtype = dtype

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        result = None
        if const_expr(param is not None):
            result = (
                utils.load_scalar_or_pointer(param, dtype=self.dtype)
                if const_expr(self.dtype is not None)
                else utils.load_scalar_or_pointer(param)
            )
        return result


class VecLoad(EpiOp):
    """Base class for broadcast vector loads (row or col) via cp_async.

    Subclasses set `dim` to 0 (M/col) or 1 (N/row) and override `_get_gmem_vec`
    for varlen handling.
    """

    dim = None  # 0 for col (M), 1 for row (N)

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    def _tile_size(self, cta_tile_shape_mnk):
        return cta_tile_shape_mnk[self.dim]

    def _broadcast_stride(self):
        # Row: stride (0,1) — broadcast along M. Col: stride (1,0) — broadcast along N.
        return (0, 1) if self.dim == 1 else (1, 0)

    def _tile_dim(self, ctx):
        return ctx.tile_N if self.dim == 1 else ctx.tile_M

    def _coord_idx(self):
        return 1 if self.dim == 1 else 0

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        if arg_tensor is None:
            return 0
        return self._tile_size(cta_tile_shape_mnk) * (arg_tensor.element_type.width // 8)

    def smem_struct_field(self, gemm, params):
        tensor = getattr(params, self.name, None)
        if tensor is None:
            size, dtype = 0, Float32
        else:
            size = self._tile_size(gemm.cta_tile_shape_mnk)
            dtype = tensor.element_type
        return (f"s_{self.name}", cute.struct.Align[cute.struct.MemRange[dtype, size], 16])

    def get_smem_tensor(self, gemm, params, storage_epi):
        if getattr(params, self.name, None) is None:
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(self._tile_size(gemm.cta_tile_shape_mnk))
        )

    def needs_async_fence(self):
        return True

    def _get_gmem_vec(self, param, ctx):
        """Get the global memory vector for this tile. Override for varlen."""
        return param[ctx.batch_idx, None]

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tDsV = None
        if const_expr(param is not None):
            dtype = param.element_type
            num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
            thr_copy = copy_utils.tiled_copy_1d(
                dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
            ).get_slice(ctx.tidx)
            mVec = self._get_gmem_vec(param, ctx)
            tile_dim = self._tile_dim(ctx)
            coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
            tVgV = thr_copy.partition_S(gVec)
            tVsV = thr_copy.partition_D(smem_tensor)
            tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
            limit = min(mVec.shape[0] - coord_idx * tile_dim, tile_dim)
            pred = cute.make_rmem_tensor((1, cute.size(tVsV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
                pred[0, m] = tVcV[0, m] < limit
            cute.copy(thr_copy, tVgV, tVsV, pred=pred)
            tDsV = ctx.partition_for_epilogue_fn(
                cute.make_tensor(
                    smem_tensor.iterator,
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
                )
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        return tDsV

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        tDrV_cvt = None
        if const_expr(state is not None):
            tDsV_cur = cute.group_modes(state, 3, cute.rank(state))[None, None, None, epi_coord]
            tDrV = cute.make_rmem_tensor(tDsV_cur.layout, tDsV_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsV_cur), cute.filter_zeros(tDrV))
            tDrV_cvt = cute.make_fragment_like(tDrV, gemm.acc_dtype)
            tDrV_cvt.store(tDrV.load().to(gemm.acc_dtype))
        return tDrV_cvt


class RowVecLoad(VecLoad):
    """Loads a row vector (N,) via cp_async, broadcasts along M with stride (0,1)."""

    dim = 1


class ColVecLoad(VecLoad):
    """Loads a col vector (M,) via cp_async, broadcasts along N with stride (1,0).
    Supports varlen_m via domain_offset."""

    dim = 0

    @cute.jit
    def _get_gmem_vec(self, param, ctx):
        if const_expr(not ctx.varlen_manager.varlen_m):
            mVec = param[ctx.batch_idx, None]
        else:
            mVec = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx],), param
            )
        return mVec

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tDsV = None
        if const_expr(param is not None):
            dtype = param.element_type
            num_copy_elems = const_expr(max(32, dtype.width)) // dtype.width
            thr_copy = copy_utils.tiled_copy_1d(
                dtype, ctx.num_epi_threads, num_copy_elems, is_async=True
            ).get_slice(ctx.tidx)
            mVec = self._get_gmem_vec(param, ctx)
            tile_dim = self._tile_dim(ctx)
            coord_idx = ctx.tile_coord_mnkl[self._coord_idx()]
            gVec = cute.local_tile(mVec, (tile_dim,), (coord_idx,))
            tVgV = thr_copy.partition_S(gVec)
            tVsV = thr_copy.partition_D(smem_tensor)
            tVcV = thr_copy.partition_S(cute.make_identity_tensor(tile_dim))
            # ColVec uses varlen-aware limit
            limit = min(
                ctx.varlen_manager.len_m(ctx.batch_idx) - coord_idx * tile_dim,
                tile_dim,
            )
            pred = cute.make_rmem_tensor((1, cute.size(tVsV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tVsV.shape[1]), unroll_full=True):
                pred[0, m] = tVcV[0, m] < limit
            cute.copy(thr_copy, tVgV, tVsV, pred=pred)
            tDsV = ctx.partition_for_epilogue_fn(
                cute.make_tensor(
                    smem_tensor.iterator,
                    cute.make_layout((ctx.tile_M, ctx.tile_N), stride=self._broadcast_stride()),
                )
            )
            if const_expr(ctx.tiled_copy_t2r is not None):
                tDsV = ctx.tiled_copy_r2s.retile(tDsV)
        return tDsV


class TileStore(EpiOp):
    """Tile-sized output tensor stored via TMA (e.g. postact).

    The params object must have: tma_atom_postact, mPostAct_mnl,
    epi_postact_smem_layout_staged, epi_tile_postact.

    Args:
        name: field name in EpilogueArguments/Params (e.g. "mPostAct")
        epi_tile_fn: optional (gemm, epi_tile) -> epi_tile for half-tile (GemmGated)
    """

    def __init__(self, name, epi_tile_fn=None):
        super().__init__(name)
        self.epi_tile_fn = epi_tile_fn

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        epi_tile = self.epi_tile_fn(gemm, gemm.epi_tile) if self.epi_tile_fn else None
        tma_atom, tma_tensor, smem_layout, epi_tile_out = setup_epi_tensor(
            gemm, tensor, epi_tile=epi_tile
        )
        return {
            "tma_atom_postact": tma_atom,
            f"{self.name}_mnl": tma_tensor,
            "epi_postact_smem_layout_staged": smem_layout,
            "epi_tile_postact": epi_tile_out,
        }

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile):
        if arg_tensor is None:
            return 0
        if self.epi_tile_fn is not None:
            epi_tile = self.epi_tile_fn(None, epi_tile)
        return cute.size(cute.shape(epi_tile)) * (arg_tensor.element_type.width // 8)

    def smem_struct_field(self, gemm, params):
        if not hasattr(gemm, "postact_dtype"):
            return (f"s_{self.name}", cute.struct.MemRange[Float32, 0])
        return (
            f"s_{self.name}",
            cute.struct.Align[
                cute.struct.MemRange[
                    gemm.postact_dtype,
                    cute.cosize(params.epi_postact_smem_layout_staged),
                ],
                gemm.buffer_align_bytes,
            ],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        if not hasattr(params, "epi_postact_smem_layout_staged"):
            return None
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            params.epi_postact_smem_layout_staged.outer,
            swizzle=params.epi_postact_smem_layout_staged.inner,
        )

    def tma_atoms(self, gemm, params):
        if hasattr(params, "tma_atom_postact"):
            return [params.tma_atom_postact]
        return []


class ColVecReduce(EpiOp):
    """Column vector reduction: accumulates across N subtiles in registers,
    then warp-reduces and writes to gmem in epi_end.

    No smem. The accumulation itself happens in epi_visit_subtile (user code).
    This op handles the register allocation (begin), per-subtile slicing (begin_loop),
    and final warp reduction + gmem write (end).
    """

    def to_params(self, gemm, args):
        return {self.name: assume_stride_divisibility(getattr(args, self.name))}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        tDrReduce = None
        if const_expr(param is not None):
            colvec_mma_layout = cute.make_layout((ctx.tile_M, ctx.tile_N), stride=(1, 0))
            tDrReduce_layout = ctx.partition_for_epilogue_fn(
                cute.make_rmem_tensor(colvec_mma_layout, Float32)
            ).layout
            tDrReduce = cute.make_rmem_tensor(tDrReduce_layout, Float32)
            cute.filter_zeros(tDrReduce).fill(0.0)
        return tDrReduce

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        result = None
        if const_expr(state is not None):
            result = cute.group_modes(state, 3, cute.rank(state))[None, None, None, epi_coord]
        return result

    @cute.jit
    def end(
        self,
        gemm,
        param,
        state,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tile_coord_mnkl,
        varlen_manager,
        tidx,
    ):
        """Warp reduction (sm90) + direct gmem write with predication."""
        if const_expr(param is not None):
            import operator
            import quack.layout_utils as layout_utils

            partition_for_epilogue_fn = partial(
                partition_for_epilogue,
                epi_tile=epi_tile,
                tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
                tidx=tidx,
                reference_src=tiled_copy_t2r is None,
            )
            tile_M, tile_N = gemm.cta_tile_shape_mnk[:2]
            tDrReduce = state
            tDrReduce_flt = cute.filter_zeros(tDrReduce)
            if const_expr(gemm.arch < 100):
                for i in cutlass.range(cute.size(tDrReduce_flt), unroll_full=True):
                    tDrReduce_flt[i] = cute.arch.warp_reduction(
                        tDrReduce_flt[i], operator.add, threads_in_group=4
                    )
            else:
                assert gemm.d_layout.is_n_major_c(), (
                    "GemmDGated only supports n-major output for now"
                )
            batch_idx = tile_coord_mnkl[3]
            limit_n = param.shape[2] if not varlen_manager.varlen_m else param.shape[1]
            if tile_coord_mnkl[1] < limit_n:
                if const_expr(not varlen_manager.varlen_m):
                    mColVec = param[batch_idx, None, tile_coord_mnkl[1]]
                else:
                    mColVec = cute.domain_offset(
                        (varlen_manager.params.cu_seqlens_m[batch_idx],),
                        param[None, tile_coord_mnkl[1]],
                    )
                gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
                limit_m = min(
                    varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M,
                    tile_M,
                )
                tDcD = partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
                tDrReduce_m = layout_utils.convert_layout_zero_stride(tDrReduce, tDrReduce.layout)[
                    None, 0
                ]
                tDcD_m = layout_utils.convert_layout_zero_stride(tDcD, tDrReduce.layout)[None, 0]
                if tDcD_m[0][1] == 0:
                    for m in cutlass.range(cute.size(tDcD_m, mode=[0])):
                        row_idx = tDcD_m[m][0]
                        if row_idx < limit_m:
                            gColVec[row_idx] = tDrReduce_m[m]
