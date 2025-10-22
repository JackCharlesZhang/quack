# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack

import torch
from torch import Tensor

import quack.utils as utils
import quack.copy_utils as copy_utils
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase
from quack.cute_dsl_utils import torch2cute_dtype_map

import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

from flash_attn.utils.torch import custom_fwd, custom_bwd
from flash_attn.utils.library import triton_op

def maybe_contiguous_lastdim(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def maybe_contiguous(x):
    return x.contiguous() if x is not None else None


def triton_autotune_configs():
    # Return configs with a valid warp count for the current device
    configs = []
    # Maximum threads per block is architecture-dependent in theory, but in reality all are 1024
    max_threads_per_block = 1024
    # Default to warp size 32 if not defined by device
    warp_size = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "warp_size", 32)
    # Autotune for warp counts which are powers of 2 and do not exceed thread per block limit
    return [triton.Config({}, num_warps=warp_count) for warp_count in [1, 2, 4, 8, 16, 32]
            if warp_count * warp_size <= max_threads_per_block]
    # return [triton.Config({}, num_warps=8)]


@triton.autotune(
    configs=triton_autotune_configs(),
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS", "HAS_DROPOUT"],
)
# torch compile doesn't like triton.heuristics, so we set these manually when calling the kernel
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
# @triton.heuristics({"HAS_ROWSCALE": lambda args: args["ROWSCALE"] is not None})
# @triton.heuristics({"HAS_DY1": lambda args: args["DY1"] is not None})
# @triton.heuristics({"HAS_DX1": lambda args: args["DX1"] is not None})
# @triton.heuristics({"HAS_B1": lambda args: args["DB1"] is not None})
# @triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    W1,
    DY1,
    DX1,
    DW1,
    DB1,
    DRESIDUAL_IN,
    ROWSCALE,
    SEEDS,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dy1_row,
    stride_dx1_row,
    stride_dres_in_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    dropout_p,
    zero_centered_weight,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_DY1: tl.constexpr,
    HAS_DX1: tl.constexpr,
    HAS_B1: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    # Do not early exit if row_start >= M, because we need to write DW and DB
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if zero_centered_weight:
        w += 1.0
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if zero_centered_weight:
            w1 += 1.0
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = (
                    tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
                )
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row).to(tl.float32)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


def _layer_norm_bwd(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    mean: Tensor,
    rstd: Tensor,
    dresidual: Optional[Tensor] = None,
    dy1: Optional[Tensor] = None,
    weight1: Optional[Tensor] = None,
    bias1: Optional[Tensor] = None,
    seeds: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[Tensor] = None,
    has_residual: bool = False,
    has_x1: bool = False,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    x_dtype: Optional[torch.dtype] = None,
    recompute_output: bool = False,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
    # Need to wrap to handle the case where dresidual_in or dx1 are aliases of x,
    # which makes torch.library unhappy
    dx, dw, db, dresidual_in, dx1, dw1, db1, y = _layer_norm_bwd_impl(
        dy,
        x,
        weight,
        bias,
        eps,
        mean,
        rstd,
        dresidual,
        dy1,
        weight1,
        bias1,
        seeds,
        dropout_p,
        rowscale,
        has_residual,
        has_x1,
        zero_centered_weight,
        is_rms_norm,
        x_dtype=x_dtype,
        recompute_output=recompute_output,
    )
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype and dropout_p == 0.0 and rowscale is None:
        dresidual_in = dx
    if has_x1 and dropout_p == 0.0:
        dx1 = dx
    return dx, dw, db, dresidual_in, dx1, dw1, db1, y



@triton_op("flash_attn::layer_norm_bwd_impl", mutates_args={},
           schema="(Tensor dy, Tensor x, Tensor weight, Tensor bias, float eps, Tensor mean, Tensor rstd, Tensor? dresidual, Tensor? dy1, Tensor? weight1, Tensor? bias1, Tensor? seeds, float dropout_p, Tensor? rowscale, bool has_residual, bool has_x1, bool zero_centered_weight, bool is_rms_norm, ScalarType? x_dtype, bool recompute_output) -> (Tensor dx, Tensor dw, Tensor db, Tensor dresidual_in, Tensor dx1, Tensor dw1, Tensor db1, Tensor y)",
           allow_decomposition=False,  # Don't let torch.compile trace inside
           )
def _layer_norm_bwd_impl(
    dy: Tensor,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    mean: Tensor,
    rstd: Tensor,
    dresidual: Optional[Tensor] = None,
    dy1: Optional[Tensor] = None,
    weight1: Optional[Tensor] = None,
    bias1: Optional[Tensor] = None,
    seeds: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[Tensor] = None,
    has_residual: bool = False,
    has_x1: bool = False,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    x_dtype: Optional[torch.dtype] = None,
    recompute_output: bool = False,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
    M, N = x.shape
    assert x.stride(-1) == 1
    dy = maybe_contiguous_lastdim(dy)
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        dresidual = maybe_contiguous_lastdim(dresidual)
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if dy1 is not None:
        dy1 = maybe_contiguous_lastdim(dy1)
        assert weight1 is not None
        assert dy1.shape == dy.shape
        assert dy1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M if not has_x1 else M * 2,)
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = (
        torch.empty_like(x)
        if has_residual
        and (dx.dtype != x.dtype or dropout_p > 0.0 or rowscale is not None or has_x1)
        else None
    )
    dx1 = torch.empty_like(dx) if (has_x1 and dropout_p > 0.0) else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None
    if recompute_output:
        assert weight1 is None, "recompute_output is not supported with parallel LayerNorm"

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # Increasing the multiple (e.g. 8) will allow more thread blocks to be launched and hide the
    # latency of the gmem reads/writes, but will increase the time of summing up dw / db.
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    _dw1 = torch.empty_like(_dw) if weight1 is not None else None
    _db1 = torch.empty_like(_db) if bias1 is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_layer_norm_bwd_kernel)[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            weight1,
            dy1,
            dx1,
            _dw1,
            _db1,
            dresidual_in,
            rowscale,
            seeds,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dy1.stride(0) if dy1 is not None else 0,
            dx1.stride(0) if dx1 is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            # Passing bool make torch inductor very unhappy since it then tries to compare to int_max
            int(zero_centered_weight),
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            bias is not None,
            dropout_p > 0.0,
            HAS_ROWSCALE=rowscale is not None,
            HAS_DY1=dy1 is not None,
            HAS_DX1=dx1 is not None,
            HAS_B1=bias1 is not None,
            RECOMPUTE_OUTPUT=y is not None,
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    dw1 = _dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    db1 = _db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    # dresidual_in and dx1 could be None, the wrapper will handle assigning them from dx
    return dx, dw, db, dresidual_in, dx1, dw1, db1, y




class RMSNorm(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        super().__init__(dtype, N, stage=1)
        self.reload_from = None if N <= 8192 else "smem"
        self.delay_w_load = False

    def _calculate_threads_per_row(self):
        """Calculate the number of threads per row for the RMSNorm kernel."""
        N = self.N
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 16384:
            return 128
        else:
            return 256

    def _set_cluster_n(self):
        """
        Set the number of clusters for the RMSNorm kernel.
        Stored in self.cluster_n.
        """
        N = self.N

        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if const_expr(self.dtype.width == 16):
            # 16-bit types (fp16, bf16)
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            # 32-bit types (fp32)
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16

        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps, dtype_res=None):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + (
                cute.size_in_bytes(dtype_res, cute.make_layout(tiler_mn))
                if dtype_res is not None
                else 0
            )
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)  # Set last dimension to be statically N
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )
        mX, mRes, mO, mResO = [
            cute.make_tensor(t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t)))
            if const_expr(t is not None)
            else None
            for t in (mX, mRes, mO, mResO)
        ]
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mRes.element_type.width if mRes is not None else 0,
                mO.element_type.width,
                mResO.element_type.width if mResO is not None else 0,
            )
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)
        if const_expr(mB is not None):
            mB_expanded_layout = cute.prepend(
                mB.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mB = cute.make_tensor(mB.iterator, mB_expanded_layout)
        if const_expr(mRstd is not None):
            mRstd_expanded_layout = cute.append(
                mRstd.layout, cute.make_layout((self.N,), stride=(0,))
            )
            mRstd = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
        self.kernel(
            mX, mW, mB, mRes, mO, mResO, mRstd, eps, tv_layout, tiler_mn, self.reload_from
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=([1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None),
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, dtype_res=mRes.element_type if mRes is not None else None
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        eps: cute.Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        mX, mRes, mO, mResO = [
            utils.domain_offset_i64((bidx * tiler_mn[0], 0), mT) if mT is not None else None
            for mT in (mX, mRes, mO, mResO)
        ]
        gX, gRes, gO, gResO = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO)
        ]
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        gRstd = (
            cute.local_tile(mRstd, tiler_mn, (bidx, cluster_y))
            if const_expr(mRstd is not None)
            else None
        )

        # declare the atoms which will be used later for memory copy
        num_copy_elems_X = tv_layout.shape[1][0]
        copy_atom_load_X_async = copy_utils.get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=True
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(
            tidx
        )

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        if const_expr(mW is not None):
            tXrW.fill(0.0)
        tXrB = cute.make_fragment_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_fragment_like(tXgRes)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        # Each copy will use the same number of elements as X and same predicate
        copy = partial(copy_utils.copy, pred=tXpX, num_copy_elems=num_copy_elems_X)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_fragment_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

        threads_per_row = tv_layout.shape[0][0]
        sum_sq_x = row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=(cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None),
        )
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(reload_from == "smem" or reload_from == "gmem"):
            if const_expr(reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
            else:
                copy(tXgX, tXrX)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                cute.autovec_copy(tXsRes, tXrRes)
                x += tXrRes.load().to(cute.Float32)
        x_hat = x * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


@torch.library.custom_op(
    "quack::_rmsnorm_fwd",
    mutates_args=("out", "rstd", "residual_out"),
    device_types="cuda",
    # We need to specify the schema manually since we're mutating an optional tensor
    schema="(Tensor x, Tensor? weight, Tensor(a2!) out, Tensor? bias, Tensor(a4!)? rstd, Tensor? residual, Tensor(a6!)? residual_out, float eps=1e-6) -> ()",
)
def _rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> None:
    """RMSNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,)
        eps: Small value for numerical stability
    Returns:
        Normalized output tensor of same shape as x
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    if weight is not None:
        assert weight.dim() == 1, "Weight must be 1D"
        assert x.shape[-1] == weight.shape[0], "Last dimension of input must match weight dimension"
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "Weight must be float32, float16 or bfloat16"
    if residual is not None:
        assert residual.shape == x.shape
        assert residual.is_cuda
        assert residual.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Residual must be float16, bfloat16, or float32"

    _, N = x.shape
    device = x.device
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    )
    x_tensor, res_tensor, out_tensor, res_out_tensor = [
        convert_from_dlpack(t) if t is not None else None for t in (x, residual, out, residual_out)
    ]
    # handle weight divisibility based on weight dtype
    if weight is not None:
        weight_dtype = torch2cute_dtype_map[weight.dtype]
        weight_tensor = utils.convert_from_dlpack(
            weight.detach(), leading_dim=0, divisibility=128 // weight_dtype.width
        )
    else:
        weight_tensor = None
    if bias is not None:
        bias_dtype = torch2cute_dtype_map[bias.dtype]
        bias_tensor = utils.convert_from_dlpack(
            bias.detach(), leading_dim=0, divisibility=128 // bias_dtype.width
        )
    else:
        bias_tensor = None
    rstd_tensor = (
        from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if rstd is not None
        else None
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (
        N,
        dtype,
        res_tensor.element_type if residual is not None else None,
        weight_tensor.element_type if weight is not None else None,
        bias_tensor.element_type if bias is not None else None,
        res_out_tensor.element_type if residual_out is not None else None,
        rstd is not None,
    )
    if compile_key not in _rmsnorm_fwd.compile_cache:
        rmsnorm_op = RMSNorm(dtype, N)
        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            rmsnorm_op,
            x_tensor,
            weight_tensor,
            bias_tensor,
            res_tensor,
            out_tensor,
            res_out_tensor,
            rstd_tensor,
            current_stream,
            eps,
        )
    _rmsnorm_fwd.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        bias_tensor,
        res_tensor,
        out_tensor,
        res_out_tensor,
        rstd_tensor,
        current_stream,
        eps,
    )


_rmsnorm_fwd.compile_cache = {}


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    # Need to wrap to handle the case where residual_out is a alias of x, which makes torch.library
    # and torch.compile unhappy. Also allocate memory for out and residual_out if they are None
    # so that _layer_norm_fwd_impl doesn't have to return them.
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32) if store_rstd else None
    if residual is not None:
        residual_dtype = residual.dtype
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None
    _rmsnorm_fwd(x, weight, out, bias, rstd, residual, residual_out, eps=eps)
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if residual_out is None:
        residual_out = x
    return out, residual_out, rstd


def rmsnorm_ref(x, w=None, bias=None, residual=None, eps=1e-6):
    x_f32 = x.float()
    if residual is not None:
        residual_f32 = residual.float()
        x_f32 += residual_f32
    x_norm = x_f32 / (torch.sqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + eps))
    out = x_norm * w if w is not None else x_norm
    if bias is not None:
        out = out + bias.float()
    if residual is None:
        return out.to(x.dtype)
    else:
        return out.to(x.dtype), x_f32.to(residual.dtype)


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    """Reference implementation for RMSNorm backward pass."""
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    if w is not None:
        wdy = dout * w
    else:
        wdy = dout
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)

    # dL/dW
    if w is not None:
        dw = (dout * x_hat).sum(dim=0)
        return dx.to(x.dtype), dw.to(w.dtype)
    else:
        return dx.to(x.dtype), None


class RMSNormBackward(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int):
        # 2 stages for double buffering when computing mean of x_hat * wdy
        super().__init__(dtype, N, stage=2, reduction_dtype=Float32)
        self.reload_wdy = None if N <= 16 * 1024 else "smem"
        if self.N > 128 * 1024 and self.dtype.width >= 32:
            # Not enough smem
            raise ValueError("RMSNormBackward does not support N > 128k with dtype >= 32 bits")

    def _get_num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 256 else (64 if N <= 512 else (128 if N <= 4096 else 256)))
            )
        )

    def _set_cluster_n(self):
        N = self.N
        cluster_n = (
            1
            if N <= 8 * 1024
            else (2 if N <= 16 * 1024 else (4 if N <= 32 * 1024 else (8 if N <= 64 * 1024 else 16)))
        )
        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps, do_dtype=None):
        if do_dtype is None:
            do_dtype = self.dtype
        return (
            # We need space for X and dO, and multiply by 2 due to double buffering
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn)) * 2
            + cute.size_in_bytes(do_dtype, cute.make_layout(tiler_mn)) * 2
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8) * 2  # mult 2 as we need 2 mbar per stage
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        semistatic_shape = (*mX.shape[:-1], self.N)  # Set last dimension to be statically N
        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )
        mX, mdO, mdResO, mdX, mdRes = [
            cute.make_tensor(t.iterator, cute.make_layout(semistatic_shape, stride=new_stride(t)))
            if const_expr(t is not None)
            else None
            for t in (mX, mdO, mdResO, mdX, mdRes)
        ]
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                mX.element_type.width,
                mdO.element_type.width,
                mdX.element_type.width,
                mdResO.element_type.width if mdResO is not None else 0,
                mdRes.element_type.width if mdRes is not None else 0,
            )
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mW is not None):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW = cute.make_tensor(mW.iterator, mW_expanded_layout)

        num_blocks = sm_count
        self.kernel(mX, mW, mdO, mdResO, mRstd, mdX, mdW, mdB, mdRes, tv_layout, tiler_mn).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps, do_dtype=mdO.element_type),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mdResO: Optional[cute.Tensor],
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: Optional[cute.Tensor],
        mdB: Optional[cute.Tensor],
        mdRes: Optional[cute.Tensor],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        shape = mX.shape
        M, N = shape[0], shape[1]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout((tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2))
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
            smem, tv_layout, is_persistent=True
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        num_copy_elems_X = tv_layout.shape[1][0]
        copy_atom_load_X = copy_utils.get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=False
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        # Each copy will use the same number of elements as X
        copy = partial(copy_utils.copy, num_copy_elems=num_copy_elems_X)

        gX, gdO, gdResO, gdX, gdRes, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdResO, mdX, mdRes, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        if const_expr(mdResO is not None):
            tXgdResO = thr_copy_X.partition_S(gdResO)
        if const_expr(mdRes is not None):
            tXgdRes = thr_copy_X.partition_D(gdRes)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0]) for thr in (tXgX, tXgdO, tXgdX)
        ]
        tXrdResO = None
        if const_expr(mdResO is not None):
            tXrdResO = cute.make_fragment_like(tXgdResO[None, None, None, 0])
        tXrdRes = None
        if const_expr(mdRes is not None):
            tXrdRes = cute.make_fragment_like(tXgdRes[None, None, None, 0])

        # This doesn't change across iterations
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
            if not is_even_N
            else None
        )

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            # Always compute partial weight gradients in fp32
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            # Always compute partial bias gradients in fp32
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE

        self._initialize_cluster(tidx, mbar_ptr, num_warps, is_persistent=True)

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            # Need this, otherwise rW can have arbitrary values that changes the reduction
            if not is_even_N:
                tXrW.fill(0.0)
            copy(tXgW, tXrW, pred=tXpX)

        # Prefetch the first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            tXgX_cur = utils.coord_offset_i64(bidx_start, tXgX, dim=3)[None, None, None, 0]
            tXgdO_cur = utils.coord_offset_i64(bidx_start, tXgdO, dim=3)[None, None, None, 0]
            copy(tXgX_cur, tXsX[None, None, None, 0], pred=tXpX, is_async=True)
            copy(tXgdO_cur, tXsdO[None, None, None, 0], pred=tXpX, is_async=True)
        elif tiler_mn[0] > 1:
            # Fill with zero, otherwise smem will be uninitialized, and we could read this back
            # later into registers, causing wrong dW.
            utils.fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
            utils.fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        threads_per_row = tv_layout.shape[0][0]
        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:  # Prefetch the next batch
                tXgX_cur = utils.coord_offset_i64(bidx + gdim, tXgX, dim=3)[None, None, None, 0]
                tXgdO_cur = utils.coord_offset_i64(bidx + gdim, tXgdO, dim=3)[None, None, None, 0]
                copy(tXgX_cur, tXsX[None, None, None, stage ^ 1], pred=tXpX, is_async=True)
                copy(tXgdO_cur, tXsdO[None, None, None, stage ^ 1], pred=tXpX, is_async=True)
            elif tiler_mn[0] > 1:
                utils.fill_oob(
                    tXsX[None, None, None, stage ^ 1],
                    None,
                    fill_value=mX.element_type.zero,
                )
                utils.fill_oob(
                    tXsdO[None, None, None, stage ^ 1],
                    None,
                    fill_value=mdO.element_type.zero,
                )
            cute.arch.cp_async_commit_group()
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            if const_expr(mdResO is not None):
                tXgdResO_cur = utils.coord_offset_i64(bidx, tXgdResO, dim=3)[None, None, None, 0]
                if row < M or tiler_mn[0] == 1:
                    copy(tXgdResO_cur, tXrdResO, pred=tXpX)
                elif tiler_mn[0] > 1:
                    tXrdResO.fill(0.0)
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            if const_expr(mdResO is not None):
                dout += tXrdResO.load().to(cute.Float32)
            x_hat = x * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                # It's faster to have 1 lane per warp to signal the mbar, rather than all lanes
                # Requires adjusting the thread_count when initializing the mbar
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                if const_expr(mdResO is not None):
                    dout += tXrdResO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                tXgdX_cur = utils.coord_offset_i64(bidx, tXgdX, dim=3)[None, None, None, 0]
                copy(tXrdX, tXgdX_cur, pred=tXpX)
            if const_expr(mdRes is not None):
                tXrdRes.store(dx.to(tXrdRes.element_type))
                tXgdRes_cur = utils.coord_offset_i64(bidx, tXgdRes, dim=3)[None, None, None, 0]
                if row < M or tiler_mn[0] == 1:
                    copy(tXrdRes, tXgdRes_cur, pred=tXpX)
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                # reduction of dw_partial within the same threadblock
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW, pred=tXpX)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy(tXrdB, tXgdB, pred=tXpX)
        else:
            # dw is already in fp32, so we can directly copy to global memory
            if const_expr(mdW is not None):
                copy(tXrdW, tXgdW, pred=tXpX)
            if const_expr(mdB is not None):
                copy(tXrdB, tXgdB, pred=tXpX)

        if const_expr(self.cluster_n > 1):  # Prevent cluster from exiting early
            # Assume state contains that next useful buffer
            # So we only need to advance to num_stages - 1 times to last used buffer
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _get_sm_count(N: int, device: torch.device) -> int:
    # This should be tuned on how many CTAs can be launched on each SM
    sm_count_multiple = (
        16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # By right, if we're using cluster, this should be cluster_count not sm_count.
    # But for cluster >= 4, due to quantization we would need to query active max cluster.
    # Instead we just do sm_count * 2, which is reasonably larger than active_cluster_count to
    # avoid wave quantization.
    sm_count = (
        sm_count * sm_count_multiple if N <= 8192 else sm_count // 2 if N <= 16384 else sm_count * 2
    )

    return sm_count


@torch.library.custom_op(
    "quack::_rmsnorm_bwd",
    mutates_args={"dx", "dw_partial", "db_partial", "dresidual"},
    device_types="cuda",
    # We need to specify the schema manually since we're mutating an optional tensor
    schema="(Tensor x, Tensor? weight, Tensor dout, Tensor rstd, Tensor(a4!) dx, Tensor(a5!)? dw_partial, Tensor(a6!)? db_partial, Tensor? dresidual_out, Tensor(a8!)? dresidual, int? sm_count) -> ()",
)
def _rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Optional[Tensor],
    db_partial: Optional[Tensor] = None,
    dresidual_out: Optional[Tensor] = None,
    dresidual: Optional[Tensor] = None,
    sm_count: Optional[int] = None,
) -> None:
    """RMSNorm backward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,)
        dout: Upstream gradients tensor of shape (M, N)
        rstd: Reciprocal standard deviation tensor of shape (M,)
    Returns:
        Tuple of (dx, dw) where:
        - dx: Input gradients tensor of same shape as x
        - dw: Weight gradients tensor of same shape as weight (or None if weight is None)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    if weight is not None:
        assert weight.dim() == 1, "Weight must be 1D"
        assert x.shape[-1] == weight.shape[0], "Last dimension of input must match weight dimension"
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in [
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ], "Weight must be float32, float16 or bfloat16"
    if dresidual_out is not None:
        assert dresidual_out.shape == x.shape
        assert dresidual_out.is_cuda
        assert dresidual_out.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Residual must be float16, bfloat16, or float32"
    if dresidual is not None:
        assert dresidual.shape == x.shape
        assert dresidual.is_cuda
        assert dresidual.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], "Residual must be float16, bfloat16, or float32"

    N = x.size(1)
    device = x.device
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
    convert_from_dlpack = lambda x: (
        from_dlpack(x.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    )
    x_tensor, dout_tensor, dres_out_tensor, dx_tensor, dres_tensor = [
        convert_from_dlpack(t) if t is not None else None
        for t in (x, dout, dresidual_out, dx, dresidual)
    ]
    # Handle weight div based on weight dtype
    if weight is not None:
        weight_dtype = torch2cute_dtype_map[weight.dtype]
        weight_tensor = utils.convert_from_dlpack(
            weight.detach(), leading_dim=0, divisibility=128 // weight_dtype.width
        )
    else:
        weight_tensor = None

    dw_partial_tensor = (
        from_dlpack(dw_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if dw_partial is not None
        else None
    )
    db_partial_tensor = (
        from_dlpack(db_partial, assumed_align=16).mark_compact_shape_dynamic(mode=0)
        if db_partial is not None
        else None
    )
    rstd_tensor = from_dlpack(rstd.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        N,
        x_tensor.element_type,
        weight_tensor.element_type if weight is not None else None,
        db_partial.dtype if db_partial is not None else None,
        dresidual.dtype if dresidual is not None else None,
        dresidual_out.dtype if dresidual_out is not None else None,
    )
    if compile_key not in _rmsnorm_bwd.compile_cache:
        rmsnorm_backward_op = RMSNormBackward(x_tensor.element_type, N)
        _rmsnorm_bwd.compile_cache[compile_key] = cute.compile(
            rmsnorm_backward_op,
            x_tensor,
            weight_tensor,
            dout_tensor,
            dres_out_tensor,
            rstd_tensor,
            dx_tensor,
            dw_partial_tensor,
            dres_tensor,
            db_partial_tensor,
            sm_count,
            current_stream,
        )

    _rmsnorm_bwd.compile_cache[compile_key](
        x_tensor,
        weight_tensor,
        dout_tensor,
        dres_out_tensor,
        rstd_tensor,
        dx_tensor,
        dw_partial_tensor,
        dres_tensor,
        db_partial_tensor,
        sm_count,
        current_stream,
    )


_rmsnorm_bwd.compile_cache = {}


def rmsnorm_bwd(
    x: Tensor,
    weight: Optional[Tensor],
    dout: Tensor,
    rstd: Tensor,
    dresidual_out: Optional[Tensor] = None,  # grad wrt residual_out
    has_bias: bool = False,
    has_residual: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    device = x.device
    N = x.size(1)
    dx = torch.empty_like(x)
    if has_residual and x.dtype != dx.dtype:
        dresidual = torch.empty_like(x, dtype=dresidual_out.dtype)
    else:
        dresidual = None
    sm_count = _get_sm_count(N, device)
    if weight is not None:
        # Always store partial gradients in fp32 for numerical accuracy
        dw_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32)
    else:
        dw_partial = None
    db_partial = torch.empty(sm_count, N, device=device, dtype=torch.float32) if has_bias else None

    _rmsnorm_bwd(
        x, weight, dout, rstd, dx, dw_partial, db_partial, dresidual_out, dresidual, sm_count
    )

    # we have summed the partial gradients in fp32, now we convert back to the weight dtype
    dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
    db = db_partial.sum(dim=0).to(weight.dtype) if has_bias else None
    # dresidual is the same as dx in this case
    if has_residual and x.dtype == dx.dtype:
        dresidual = dx
    return dx, dw, db, dresidual


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual=None,
        out_dtype=None,
        residual_dtype=None,
        eps=1e-6,
        prenorm=False,
    ):
        x_shape_og = x.shape
        # Flatten input
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1])
        need_grad = any(ctx.needs_input_grad[:3])
        out, residual_out, rstd = rmsnorm_fwd(
            x,
            weight,
            bias=bias,
            residual=residual,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            eps=eps,
            store_rstd=need_grad,
        )
        ctx.save_for_backward(x if residual is None else residual_out, weight, rstd, bias) # JCZ added bias here
        ctx.has_bias = bias is not None
        ctx.eps = eps
        ctx.x_shape_og = x_shape_og
        ctx.residual_dtype = residual.dtype if residual is not None else None
        ctx.prenorm = prenorm
        ctx.has_residual = residual is not None
        if residual_out is None or not prenorm:
            return out.reshape(x_shape_og)
        else:
            return out.reshape(x_shape_og), residual_out.reshape(x_shape_og)

    # @staticmethod
    # def backward(ctx, dout, *args):
    #     x, weight, rstd = ctx.saved_tensors
    #     has_bias = ctx.has_bias
    #     has_residual = ctx.has_residual
    #     if ctx.prenorm and ctx.residual_dtype is not None:
    #         dresidual_out = args[0]
    #         dresidual_out = dresidual_out.reshape(-1, dresidual_out.shape[-1])
    #     else:
    #         dresidual_out = None
    #     x_shape_og = ctx.x_shape_og
    #     # Reshape dout to match the flattened shape used in forward
    #     dout = dout.view(-1, dout.shape[-1])

    #     dx, dw, db, dresidual = rmsnorm_bwd(x, weight, dout, rstd, dresidual_out, has_bias, has_residual)
    #     dx = dx.view(x_shape_og)
    #     if dresidual is not None:
    #         dresidual = dresidual.reshape(x_shape_og)

    #     return dx, dw, db, dresidual, *([None] * 4)

    # TRITON
    @staticmethod
    def backward(ctx, dy, *args):
        # x, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd = ctx.saved_tensors
        x, weight, rstd, bias = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        # if weight1 is not None:
        #     dy1, args = args[0], args[1:]
        #     dy1 = dy1.reshape(-1, dy1.shape[-1])
        #     assert dy1.shape == x.shape
        # else:
        dy1 = None
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in, _, _, _, _ = _layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            None, # mean
            rstd,
            dresidual,
            dy1,
            None,
            None, # bias1
            None, # seeds
            0.0, # dropout
            None, # rowscale
            ctx.has_residual,
            False, # has_x1
            True, # zero_centered_weight
            True, # is_rms_norm
            x.dtype,
            recompute_output=False,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None
        )



def rmsnorm(
    x: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
) -> Tensor:
    """RMSNorm with automatic differentiation support.

    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,)
        eps: Small value for numerical stability

    Returns:
        Normalized output tensor of same shape as x
    """
    return RMSNormFunction.apply(x, weight, bias, residual, out_dtype, residual_dtype, eps, prenorm)


class QuackRMSNorm(torch.nn.RMSNorm):
    """RMSNorm module that behaves like torch.nn.RMSNorm.

    This class provides a drop-in replacement for torch.nn.RMSNorm that uses
    the quack.rmsnorm implementation under the hood.

    Args:
        dim (int): The dimension to normalize over
        eps (float, optional): A small constant for numerical stability. Default: 1e-6

    Attributes:
        weight (torch.nn.Parameter): The learnable weight parameter
        eps (float): A small constant for numerical stability
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True, device=None, dtype=None
    ):
        super().__init__(dim, eps, elementwise_affine, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm to the input tensor.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized tensor
        """
        return rmsnorm(x, self.weight, eps=self.eps)
