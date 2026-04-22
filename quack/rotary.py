# Copyright (c) 2026, Tri Dao.

import math
from functools import partial
from typing import Optional, Union

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import copy_utils
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


def _ensure_last_dim_contiguous(t: Tensor) -> Tensor:
    """Ensure last-dim stride is 1 while avoiding copies for strided-but-row-contiguous inputs."""
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.stride(-1) == 1 else t.contiguous()


class RotaryVectorized:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        dim: int,
        interleaved: bool = False,
        conjugate: bool = False,
    ):
        self.dtype = dtype
        self.dim = dim
        self.interleaved = interleaved
        self.conjugate = conjugate
        self.num_threads = 128
        self.tile_h = 2 if self.dim <= 64 else 1
        multiple = 32 if dim <= 32 else 64
        self.tile_d = (dim + multiple - 1) // multiple * multiple

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mSeqlenOffsets: Optional[cute.Tensor],
        mCuSeqlens: Optional[cute.Tensor],
        mO: cute.Tensor,
        seqlen_offsets: Int32,
        max_seqlen: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        assert mCos.element_type == mSin.element_type
        assert mCos.shape[1] == mSin.shape[1]
        assert mCos.shape[1] * 2 == self.dim

        self.is_varlen = const_expr(mCuSeqlens is not None)

        # If not self.interleaved, we use cp.async for copying X from gmem -> smem, sync, then
        # smem -> rmem with a different thread layout.
        # If self.interleaved, then we directly copy X from gmem -> rmem, so the layout for X
        # has to be compatible w the layout for cos/sin.
        vecsize = math.gcd(128 // mX.element_type.width, self.dim)
        if const_expr(not self.interleaved):
            vecsize_cs = math.gcd(128 // mCos.element_type.width, self.dim // 2)
        else:
            vecsize_cs = vecsize // 2
            assert (128 // mCos.element_type.width) % vecsize_cs == 0
        vecs_per_row = self.tile_d // vecsize
        vecs_per_row_cs = self.tile_d // 2 // vecsize_cs
        threads_per_row = math.gcd(32, vecs_per_row)
        threads_per_row_cs = math.gcd(32, vecs_per_row_cs)
        if const_expr(not self.interleaved):
            # Multiply so that all threads can fetch 1 cos and 1 sin.
            multiple = max(mX.element_type.width * 2 // mCos.element_type.width, 1)
        else:
            multiple = 1
        tiler_mn = (self.num_threads // threads_per_row * multiple, self.tile_d)
        tiled_copy = copy_utils.tiled_copy_2d(
            mX.element_type, threads_per_row, self.num_threads, vecsize
        )
        tiled_copy_cs = copy_utils.tiled_copy_2d(
            mCos.element_type, threads_per_row_cs, self.num_threads, vecsize_cs
        )
        assert tiler_mn[0] % (self.num_threads // threads_per_row_cs) == 0

        # (b, s, h, d) -> (s, d, h, b); (s, h, d) -> (s, d, h)
        x_layout_transpose = [0, 2, 1] if const_expr(self.is_varlen) else [1, 3, 2, 0]
        mX, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=x_layout_transpose))
            for t in (mX, mO)
        ]
        assert cute.rank(mX) == (3 if const_expr(self.is_varlen) else 4)
        if const_expr(self.is_varlen):
            batch = mCuSeqlens.shape[0] - 1
            seqlen = max_seqlen
            nheads = mX.shape[2]
        else:
            batch = mX.shape[3]
            seqlen = mX.shape[0]
            nheads = mX.shape[2]
        self.kernel(
            mX,
            mCos,
            mSin,
            mSeqlenOffsets,
            mCuSeqlens,
            mO,
            seqlen_offsets,
            max_seqlen,
            tiler_mn,
            tiled_copy,
            tiled_copy_cs,
        ).launch(
            grid=[
                cute.ceil_div(nheads, self.tile_h),
                cute.ceil_div(seqlen, tiler_mn[0]),
                batch,
            ],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mSeqlenOffsets: Optional[cute.Tensor],
        mCuSeqlens: Optional[cute.Tensor],
        mO: cute.Tensor,
        seqlen_offsets: Int32,
        max_seqlen: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        tiled_copy_cs: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_idx, m_idx, batch_idx = cute.arch.block_idx()

        tiler_mnh = (tiler_mn[0], tiler_mn[1], self.tile_h)
        tiler_cossin = (tiler_mn[0], tiler_mn[1] // 2)

        smem = cutlass.utils.SmemAllocator()
        sX = None
        if const_expr(not self.interleaved):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mnh, order=(1, 0, 2)),
                byte_alignment=16,
            )

        offset = seqlen_offsets
        if const_expr(mSeqlenOffsets is not None):
            offset = Int32(mSeqlenOffsets[batch_idx])

        seq_len = max_seqlen
        cX_cols = const_expr(max(self.dim, tiler_mn[1]))
        if const_expr(self.is_varlen):
            seq_start = Int32(mCuSeqlens[batch_idx])
            seq_len = Int32(mCuSeqlens[batch_idx + 1]) - seq_start
            nheads = mX.shape[2]
            cX_shape = (max_seqlen, cX_cols)
        else:
            seq_len = mX.shape[0]
            nheads = mX.shape[2]
            cX_shape = (mX.shape[0], cX_cols)

        if const_expr(self.is_varlen):
            mX_batch = cute.domain_offset((seq_start, None, None), mX)
            mO_batch = cute.domain_offset((seq_start, None, None), mO)
        else:
            mX_batch = mX[None, None, None, batch_idx]
            mO_batch = mO[None, None, None, batch_idx]
        gX = cute.local_tile(mX_batch, tiler_mnh, (m_idx, 0, head_idx))
        gO = cute.local_tile(mO_batch, tiler_mnh, (m_idx, 0, head_idx))
        gCos = cute.local_tile(cute.domain_offset((offset, None), mCos), tiler_cossin, (m_idx, 0))
        gSin = cute.local_tile(cute.domain_offset((offset, None), mSin), tiler_cossin, (m_idx, 0))
        cX = cute.local_tile(cute.make_identity_tensor(cX_shape), tiler_mn, (m_idx, 0))
        cCosSin = cute.local_tile(
            cute.make_identity_tensor((max_seqlen, mCos.shape[1])), tiler_cossin, (m_idx, 0)
        )

        thr_copy = tiled_copy.get_slice(tidx)
        tXcX_full = thr_copy.partition_S(cX)
        thr_copy_cs = tiled_copy_cs.get_slice(tidx)
        tScCosSin_full = thr_copy_cs.partition_S(cCosSin)
        tXcX = tXcX_full[(0, None), None, None]
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX) if const_expr(sX is not None) else None
        tXgO = thr_copy.partition_D(gO)
        tCSgCos = thr_copy_cs.partition_S(gCos)
        tCSgSin = thr_copy_cs.partition_S(gSin)
        tCSrCos = cute.make_rmem_tensor_like(tCSgCos)
        tCSrSin = cute.make_rmem_tensor_like(tCSgSin)
        tXrX_g2r = cute.make_rmem_tensor_like(tXgX)

        is_even_dim = const_expr(tiler_mn[1] == self.dim)
        pred, pred_cs = None, None
        if const_expr(not is_even_dim):
            pred = copy_utils.predicate_k(tXcX_full, limit=self.dim)
            pred_cs = copy_utils.predicate_k(tScCosSin_full, limit=self.dim // 2)
        copy = partial(copy_utils.copy, pred=pred[None, 0, None] if not is_even_dim else None)
        copy_cs = partial(copy_utils.copy, pred=pred_cs[None, 0, None] if not is_even_dim else None)

        tScCosSin = tScCosSin_full[(0, None), None, None]
        for m in cutlass.range(cute.size(tCSgCos, mode=[1]), unroll_full=True):
            row_cs = tScCosSin[0, m, 0][0]
            sincos_is_valid = row_cs < seq_len
            if const_expr(mSeqlenOffsets is not None):
                sincos_is_valid = sincos_is_valid and row_cs + offset < mCos.shape[0]
            if sincos_is_valid:
                copy_cs(tCSgCos[None, m, None], tCSrCos[None, m, None])
                copy_cs(tCSgSin[None, m, None], tCSrSin[None, m, None])

        for h in cutlass.range_constexpr(self.tile_h):
            if self.tile_h == 1 or h < nheads - head_idx * self.tile_h:
                for m in cutlass.range(cute.size(tXgX, mode=[1]), unroll_full=True):
                    if tXcX[0, m, 0][0] < seq_len:
                        if const_expr(not self.interleaved):
                            copy(tXgX[None, m, None, h], tXsX[None, m, None, h], is_async=True)
                        else:
                            copy(tXgX[None, m, None, h], tXrX_g2r[None, m, None, h])
            if const_expr(not self.interleaved):
                cute.arch.cp_async_commit_group()

        cos_vals = tCSrCos.load().to(Float32)
        sin_vals = tCSrSin.load().to(Float32)
        if const_expr(self.conjugate):
            sin_vals = -sin_vals
        rCos = cute.make_rmem_tensor(tCSrCos.shape, Float32)
        rCos.store(cos_vals)
        rSin = cute.make_rmem_tensor(tCSrSin.shape, Float32)
        rSin.store(sin_vals)
        if const_expr(not self.interleaved):
            sX0 = cute.composition(sX, (tiler_mn[0], tiler_mn[1] // 2, self.tile_h))
            sX1 = cute.domain_offset((None, self.dim // 2, None), sX0)
            tCsX0 = thr_copy_cs.partition_D(sX0)
            tCsX1 = thr_copy_cs.partition_D(sX1)
            for h in cutlass.range_constexpr(self.tile_h):
                cute.arch.cp_async_wait_group(self.tile_h - h - 1)
                cute.arch.sync_threads()
                tCrX0 = copy_utils.load_s2r(tCsX0[None, None, None, h])
                tCrX1 = copy_utils.load_s2r(tCsX1[None, None, None, h])
                x0_vals = tCrX0.load().to(Float32)
                x1_vals = tCrX1.load().to(Float32)
                tCrX0.store((x0_vals * cos_vals - x1_vals * sin_vals).to(tCrX0.element_type))
                tCrX1.store((x0_vals * sin_vals + x1_vals * cos_vals).to(tCrX1.element_type))
                if const_expr(is_even_dim):
                    cute.autovec_copy(tCrX0, tCsX0[None, None, None, h])
                else:
                    for k in cutlass.range(cute.size(tCrX0, mode=[2]), unroll_full=True):
                        if pred_cs[0, 0, k]:  # Need predication to avoid overwriting tCsX1
                            cute.autovec_copy(tCrX0[None, None, k], tCsX0[None, None, k, h])
                cute.autovec_copy(tCrX1, tCsX1[None, None, None, h])
            cute.arch.sync_threads()
        else:
            for h in cutlass.range_constexpr(self.tile_h):
                tCrX_f32 = cute.make_rmem_tensor(tXrX_g2r[None, None, None, 0].shape, Float32)
                tCrX_f32.store(tXrX_g2r[None, None, None, h].load().to(Float32))
                assert cute.size(tCrX_f32.shape) == cute.size(rCos) * 2
                for i in cutlass.range(cute.size(tCrX_f32.shape) // 2, unroll_full=True):
                    x0, x1 = tCrX_f32[2 * i], tCrX_f32[2 * i + 1]
                    tCrX_f32[2 * i] = x0 * rCos[i] - x1 * rSin[i]
                    tCrX_f32[2 * i + 1] = x0 * rSin[i] + x1 * rCos[i]
                tXrX_g2r[None, None, None, h].store(tCrX_f32.load().to(tXrX_g2r.element_type))

        for h in cutlass.range_constexpr(self.tile_h):
            if const_expr(not self.interleaved):
                tXrX = copy_utils.load_s2r(tXsX[None, None, None, h])
            else:
                tXrX = tXrX_g2r[None, None, None, h]
            if self.tile_h == 1 or h < nheads - head_idx * self.tile_h:
                for m in cutlass.range(cute.size(tXgO, mode=[1]), unroll_full=True):
                    if tXcX[0, m, 0][0] < seq_len:
                        copy(tXrX[None, m, None], tXgO[None, m, None, h])


@jit_cache
def _compile_rotary(
    dtype,
    cossin_dtype,
    seqlen_offsets_dtype,
    cu_seqlens_dtype,
    dim,
    interleaved,
    conjugate,
):
    is_varlen = cu_seqlens_dtype is not None
    has_seqlen_offsets_tensor = seqlen_offsets_dtype is not None
    batch_sym = cute.sym_int()
    batch_p1_sym = cute.sym_int()
    seqlen_sym = cute.sym_int()
    total_seqlen_sym = cute.sym_int()
    nheads_sym = cute.sym_int()
    x_dim_sym = cute.sym_int()
    seqlen_ro_sym = cute.sym_int()
    x_shape = (
        (total_seqlen_sym, nheads_sym, x_dim_sym)
        if is_varlen
        else (batch_sym, seqlen_sym, nheads_sym, x_dim_sym)
    )
    x_divby = math.gcd(128 // dtype.width, dim)
    cossin_divby = math.gcd(128 // cossin_dtype.width, dim // 2)
    x_cute = fake_tensor(dtype, x_shape, x_divby)
    out_cute = fake_tensor(dtype, x_shape, x_divby)
    cos_cute = fake_tensor(cossin_dtype, (seqlen_ro_sym, dim // 2), cossin_divby)
    sin_cute = fake_tensor(cossin_dtype, (seqlen_ro_sym, dim // 2), cossin_divby)
    seqlen_offsets_cute = (
        fake_tensor(seqlen_offsets_dtype, (batch_sym,)) if has_seqlen_offsets_tensor else None
    )
    cu_seqlens_cute = fake_tensor(cu_seqlens_dtype, (batch_p1_sym,)) if is_varlen else None
    return cute.compile(
        RotaryVectorized(dtype, dim, interleaved=interleaved, conjugate=conjugate),
        x_cute,
        cos_cute,
        sin_cute,
        seqlen_offsets_cute,
        cu_seqlens_cute,
        out_cute,
        Int32(0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _launch_rotary(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets_tensor: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    out: Tensor,
    seqlen_offsets: int,
    max_seqlen: int,
    interleaved: bool,
    conjugate: bool,
) -> None:
    assert x.stride(-1) == 1 and out.stride(-1) == 1, (
        "Rotary vectorized path requires last-dim stride 1"
    )
    assert cos.dtype == sin.dtype and cos.shape == sin.shape
    dtype = torch2cute_dtype_map[x.dtype]
    cossin_dtype = torch2cute_dtype_map[cos.dtype]
    dim_half = cos.size(1)
    dim = dim_half * 2
    seqlen_offsets_dtype = (
        torch2cute_dtype_map[seqlen_offsets_tensor.dtype]
        if seqlen_offsets_tensor is not None
        else None
    )
    cu_seqlens_dtype = Int32 if cu_seqlens is not None else None
    _compile_rotary(
        dtype,
        cossin_dtype,
        seqlen_offsets_dtype,
        cu_seqlens_dtype,
        dim,
        interleaved,
        conjugate,
    )(x, cos, sin, seqlen_offsets_tensor, cu_seqlens, out, seqlen_offsets, max_seqlen)


@torch.library.custom_op(
    "quack::_rotary_fwd_out",
    mutates_args=("out",),
    device_types="cuda",
    schema="(Tensor x, Tensor cos, Tensor sin, Tensor? seqlen_offsets_tensor, Tensor? cu_seqlens, Tensor(a!) out, int seqlen_offsets, int max_seqlen, bool interleaved, bool conjugate) -> ()",
)
def _rotary_fwd_out(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets_tensor: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    out: Tensor,
    seqlen_offsets: int,
    max_seqlen: int,
    interleaved: bool,
    conjugate: bool,
) -> None:
    _launch_rotary(
        x,
        cos,
        sin,
        seqlen_offsets_tensor,
        cu_seqlens,
        out,
        seqlen_offsets,
        max_seqlen,
        interleaved,
        conjugate,
    )


@_rotary_fwd_out.register_fake
def _rotary_fwd_out_fake(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets_tensor: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    out: Tensor,
    seqlen_offsets: int,
    max_seqlen: int,
    interleaved: bool,
    conjugate: bool,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(cos.size(1), torch.SymInt):
        _launch_rotary(
            x,
            cos,
            sin,
            seqlen_offsets_tensor,
            cu_seqlens,
            out,
            seqlen_offsets,
            max_seqlen,
            interleaved,
            conjugate,
        )


@torch.library.custom_op(
    "quack::_rotary_fwd_inplace",
    mutates_args=("x",),
    device_types="cuda",
    schema="(Tensor(a!) x, Tensor cos, Tensor sin, Tensor? seqlen_offsets_tensor, Tensor? cu_seqlens, int seqlen_offsets, int max_seqlen, bool interleaved, bool conjugate) -> ()",
)
def _rotary_fwd_inplace(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets_tensor: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    seqlen_offsets: int,
    max_seqlen: int,
    interleaved: bool,
    conjugate: bool,
) -> None:
    _launch_rotary(
        x,
        cos,
        sin,
        seqlen_offsets_tensor,
        cu_seqlens,
        x,
        seqlen_offsets,
        max_seqlen,
        interleaved,
        conjugate,
    )


@_rotary_fwd_inplace.register_fake
def _rotary_fwd_inplace_fake(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets_tensor: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    seqlen_offsets: int,
    max_seqlen: int,
    interleaved: bool,
    conjugate: bool,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(cos.size(1), torch.SymInt):
        _launch_rotary(
            x,
            cos,
            sin,
            seqlen_offsets_tensor,
            cu_seqlens,
            x,
            seqlen_offsets,
            max_seqlen,
            interleaved,
            conjugate,
        )


def _normalize_seqlen_offsets(
    seqlen_offsets: Union[int, Tensor], batch: int
) -> tuple[int, Optional[Tensor]]:
    if isinstance(seqlen_offsets, Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        return 0, _ensure_last_dim_contiguous(seqlen_offsets)
    return int(seqlen_offsets), None


def apply_rotary(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> Tensor:
    """
    Apply rotary embedding to the first rotary_dim dimensions of x.

    x is (batch, seqlen, nheads, headdim) when cu_seqlens is None, otherwise
    (total_seqlen, nheads, headdim). cos/sin are (seqlen_ro, rotary_dim / 2).
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported x dtype"
    assert cos.dtype == sin.dtype, "cos and sin must have the same dtype"
    assert cos.dtype in supported_types and sin.dtype in supported_types, (
        "Unsupported cos/sin dtype"
    )
    assert sin.shape == cos.shape
    assert cos.dim() == 2
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        assert x.dim() == 4
        batch, seqlen, _nheads, headdim = x.shape
        max_seqlen = seqlen
    else:
        assert x.dim() == 3
        assert max_seqlen is not None, "If cu_seqlens is passed, max_seqlen must be passed"
        assert cu_seqlens.dtype == torch.int32, "cu_seqlens must have dtype torch.int32"
        total_seqlen, _nheads, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert rotary_dim <= 512, "Only support rotary_dim <= 512"
    assert headdim <= 512, "Only support headdim <= 512"
    assert headdim % 8 == 0, "headdim must be divisible by 8"
    assert rotary_dim % 8 == 0, "rotary_dim must be divisible by 8"
    assert rotary_dim % 2 == 0
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"
    offset_int, seqlen_offsets_tensor = _normalize_seqlen_offsets(seqlen_offsets, batch)
    if seqlen_offsets_tensor is None:
        assert offset_int + seqlen <= seqlen_ro

    cos, sin = _ensure_last_dim_contiguous(cos), _ensure_last_dim_contiguous(sin)
    out = x if inplace else torch.empty_like(x)
    if rotary_dim < headdim and not inplace:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])
    if inplace:
        _rotary_fwd_inplace(
            x,
            cos,
            sin,
            seqlen_offsets_tensor,
            cu_seqlens,
            offset_int,
            int(max_seqlen),
            interleaved,
            conjugate,
        )
    else:
        _rotary_fwd_out(
            x,
            cos,
            sin,
            seqlen_offsets_tensor,
            cu_seqlens,
            out,
            offset_int,
            int(max_seqlen),
            interleaved,
            conjugate,
        )
    return out


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleaved: bool = False,
        inplace: bool = False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=False,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


apply_rotary_emb_func = apply_rotary_emb


def apply_rotary_emb_qkv_(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    cos_k: Optional[Tensor] = None,
    sin_k: Optional[Tensor] = None,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
) -> Tensor:
    apply_rotary_fn = partial(
        apply_rotary_emb,
        interleaved=interleaved,
        inplace=True,
        seqlen_offsets=seqlen_offsets,
    )
    if cos_k is None and sin_k is None and qkv.is_contiguous():
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            qk = apply_rotary_fn(qk, cos, sin)
            return torch.cat([qk.reshape(batch, seqlen, 2, nheads, headdim), qkv[:, :, 2:]], dim=2)
        assert qkv.dim() == 4
        assert num_heads_q is not None
        num_heads_k = (qkv.shape[2] - num_heads_q) // 2
        assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
        qk = qkv[:, :, : num_heads_q + num_heads_k]
        qk = apply_rotary_fn(qk, cos, sin)
        return torch.cat([qk, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)

    cos_k = cos if cos_k is None else cos_k
    sin_k = sin if sin_k is None else sin_k
    if qkv.dim() == 5:
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        q, k = qkv[:, :, 0], qkv[:, :, 1]
        q = apply_rotary_fn(q, cos, sin)
        k = apply_rotary_fn(k, cos_k, sin_k)
        return torch.stack([q, k, qkv[:, :, 2]], dim=2)

    assert qkv.dim() == 4
    assert num_heads_q is not None
    num_heads_k = (qkv.shape[2] - num_heads_q) // 2
    assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
    q, k = qkv[:, :, :num_heads_q], qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
    q = apply_rotary_fn(q, cos, sin)
    k = apply_rotary_fn(k, cos_k, sin_k)
    return torch.cat([q, k, qkv[:, :, num_heads_q + num_heads_k :]], dim=2)


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleaved: bool = False,
        seqlen_offsets: Union[int, Tensor] = 0,
    ):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        apply_rotary(
            kv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        dk = apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=False,
            conjugate=True,
        )
        dkv = torch.stack([dk, dkv[:, :, 1]], dim=2)
        return dkv, None, None, None, None


def apply_rotary_emb_kv_(
    kv: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
) -> Tensor:
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)
