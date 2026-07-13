# Copyright (c) 2026, Tri Dao.
"""Per-head RMSNorm resources for composed GEMM epilogues."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import quack.copy_utils as copy_utils
from quack.epi_ops import EpiOp


class HeadRMSNormStats(EpiOp):
    """Per-head RMSNorm statistics and scale resource:
    prepass sink (fn_sink_flush: smem-atomic per-(row, head) sums of the
    squared values the prepass fn returns) + main-phase value port
    (fn_prepare: dense per-element rsqrt(mean + eps) * w multiplier)."""

    fn_port = "value"

    def __init__(self, name, eps=1e-6):
        super().__init__(name)
        self.eps = eps

    def config_key(self):
        return (self.eps,)

    def host_arg_key(self, value):
        from quack.cute_dsl_utils import torch2cute_dtype_map

        return (torch2cute_dtype_map[value.dtype], value.shape[0])

    def host_fake_arg(self, key, fctx):
        from quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(dtype, (head_dim,), leading_dim=0, divisibility=128 // dtype.width)

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        tensor = getattr(args, self.name)
        # One accumulator per (tile row, head). Sizing by tile_M (not
        # epi_tile[0]) matters on SM90, whose epi tiles are 64 rows: epi_M > 1
        # subtiles land on distinct rows and must not alias.
        rows = gemm.cta_tile_shape_mnk[0]
        heads = gemm.cta_tile_shape_mnk[1] // tensor.shape[0]
        gemm._head_rmsnorm_smem_shape = (rows, heads)
        return {self.name: tensor}

    def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None):
        from quack.epi_ops import EpiSmemBytes
        from cutlass import Float32

        rows = cta_tile_shape_mnk[0]
        heads = cta_tile_shape_mnk[1] // arg_tensor.shape[0]
        return EpiSmemBytes(unstaged=rows * heads * (Float32.width // 8))

    def smem_struct_field(self, gemm, params):
        from cutlass import Float32

        rows, heads = gemm._head_rmsnorm_smem_shape
        return (
            f"s_{self.name}",
            cute.struct.Align[cute.struct.MemRange[Float32, rows * heads], 16],
        )

    def get_smem_tensor(self, gemm, params, storage_epi):
        return getattr(storage_epi, f"s_{self.name}").get_tensor(
            cute.make_layout(gemm._head_rmsnorm_smem_shape)
        )

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        from cutlass import Float32

        assert gemm.arch in (90, 100), "head RMSNorm stats need the acc prepass (SM90/SM100)"
        head_dim = const_expr(param.shape[0])
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        assert tile_N % head_dim == 0, "head RMSNorm needs whole heads per tile"
        gW = cute.make_tensor(
            param.iterator,
            cute.make_layout(
                (tile_M, (head_dim, tile_N // head_dim)), stride=(0, (param.stride[0], 0))
            ),
        )
        tDgW = ctx.partition_for_epilogue_fn(gW)
        tDcW = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
        tDrM_ref = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(cute.make_layout((tile_M, tile_N), stride=(1, 0)), Float32)
        )
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDgW = ctx.tiled_copy_r2s.retile(tDgW)
            tDcW = ctx.tiled_copy_r2s.retile(tDcW)
            tDrM_ref = ctx.tiled_copy_r2s.retile(tDrM_ref)
        tDgW = cute.group_modes(tDgW, 3, cute.rank(tDgW))
        tDcW = cute.group_modes(tDcW, 3, cute.rank(tDcW))
        ref_layout = cute.group_modes(tDrM_ref, 3, cute.rank(tDrM_ref))[None, None, None, 0].layout
        wbuf_raw = cute.make_rmem_tensor_like(tDgW[None, None, None, 0])
        copy_vec = const_expr(
            min(
                cute.max_common_vector(
                    cute.coalesce(cute.filter_zeros(tDgW[None, None, None, 0])),
                    cute.coalesce(cute.filter_zeros(wbuf_raw)),
                ),
                128 // param.element_type.width,
            )
        )
        if const_expr(param.element_type != Float32):
            wbuf_cvt = cute.make_rmem_tensor(wbuf_raw.shape, Float32)
        else:
            wbuf_cvt = wbuf_raw
        # Zero the (row, head) accumulators here: begin runs before the driver
        # prepass sweep, so no per-subtile epi_idx==0 special case is needed.
        # Strided over rows: tile_M can exceed the epilogue thread count
        # (e.g. 192-row tiles under pingpong's single 128-thread warpgroup,
        # whose exclusive epilogue window covers the shared smem).
        rows, num_heads = const_expr(smem_tensor.shape[0]), const_expr(smem_tensor.shape[1])
        num_epi_threads = const_expr(getattr(gemm, "num_epi_warps", 4) * 32)
        for r0 in cutlass.range_constexpr(0, rows, num_epi_threads):
            r = r0 + ctx.tidx
            if r < rows:
                for h in cutlass.range_constexpr(num_heads):
                    smem_tensor[r, h] = Float32(0.0)
        ctx.epilogue_barrier.arrive_and_wait()
        return [smem_tensor, tDgW, tDcW, ref_layout, head_dim, wbuf_raw, wbuf_cvt, copy_vec]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        from cutlass import Float32

        smem_tensor, tDgW, tDcW, ref_layout, head_dim, wbuf_raw, wbuf_cvt, copy_vec = state
        tDgW_cur = tDgW[None, None, None, epi_coord]
        tiler = cute.make_layout(copy_vec)
        cute.copy(
            copy_utils.get_copy_atom(wbuf_raw.element_type, copy_vec),
            cute.zipped_divide(cute.coalesce(cute.filter_zeros(tDgW_cur)), tiler),
            cute.zipped_divide(cute.coalesce(cute.filter_zeros(wbuf_raw)), tiler),
        )
        if const_expr(wbuf_cvt is not wbuf_raw):
            wbuf_cvt.store(wbuf_raw.load().to(Float32))
        tDcW_cur = tDcW[None, None, None, epi_coord]
        return [smem_tensor, wbuf_cvt, tDcW_cur, ref_layout, head_dim]

    @cute.jit
    def fn_sink_flush(self, gemm, state, frag):
        """Prepass sink: frag holds the squared values; combine per-thread
        row-run partials with one smem atomic per (row, head)."""
        from cutlass import Float32
        import quack.layout_utils as layout_utils

        sSum, _, tDcW_cur, ref_layout, head_dim = state
        x_mn = layout_utils.convert_layout_zero_stride(frag, ref_layout)
        c_mn = layout_utils.convert_layout_zero_stride(tDcW_cur, ref_layout)
        num_rows = const_expr(cute.size(x_mn, mode=[0]))
        num_cols = const_expr(cute.size(x_mn, mode=[1]))
        assert head_dim % num_cols == 0, "thread column run must divide head_dim"
        for r in cutlass.range_constexpr(num_rows):
            partial = Float32(0.0)
            for c in cutlass.range(num_cols, unroll_full=True):
                partial += x_mn[r, c]
            # coord[0] is the tile-M row (identity tensor over the full CTA
            # tile), indexing the (tile_M, heads) accumulator directly.
            coord = c_mn[r, 0]
            cute.arch.atomic_add(
                sSum.iterator + cute.crd2idx((coord[0], coord[1] // head_dim), sSum.layout),
                partial,
            )

    @cute.jit
    def fn_prepare(self, gemm, state, paired):
        """Main-phase value: dense per-element rsqrt(mean + eps) * w multiplier."""
        from cutlass import Float32
        import quack.layout_utils as layout_utils

        sSum, wfrag, tDcW_cur, ref_layout, head_dim = state
        inv_d = const_expr(1.0 / head_dim)
        out = cute.make_rmem_tensor(wfrag.layout.shape, Float32)
        out_mn = layout_utils.convert_layout_zero_stride(out, ref_layout)
        w_mn = layout_utils.convert_layout_zero_stride(wfrag, ref_layout)
        c_mn = layout_utils.convert_layout_zero_stride(tDcW_cur, ref_layout)
        for r in cutlass.range_constexpr(cute.size(out_mn, mode=[0])):
            coord = c_mn[r, 0]
            total = sSum[coord[0], coord[1] // head_dim]
            rstd = cute.math.rsqrt(total * inv_d + Float32(self.eps), fastmath=True)
            for c in cutlass.range(cute.size(out_mn, mode=[1]), unroll_full=True):
                out_mn[r, c] = rstd * w_mn[r, c]
        return out
