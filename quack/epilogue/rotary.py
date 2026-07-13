# Copyright (c) 2026, Han Guo, Tri Dao.
"""Rotary-position resources and ready-to-use RoPE GEMM epilogues."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, const_expr

import quack.copy_utils as copy_utils
from quack.epi_ops import EpiOp, TileLoad
from quack.gemm_epilogue import gemm_epilogue, pack, unpack


class RotaryCosSinLoad(EpiOp):
    """Per-subtile gmem->rmem load of the interleaved rotary cos/sin table.

    The param tensor is (seqlen_ro, head_dim), row-major, cos at even columns
    and sin at odd columns. Output column n reads table column n % head_dim of
    table row m: the head broadcast is expressed as a stride-0 repeat mode in
    the per-tile gmem view, so a single ``partition_for_epilogue`` makes the
    loaded fragment elementwise-aligned with tRS_rD.

    Requires tile_N % head_dim == 0 or head_dim % tile_N == 0 (the repeat /
    slice views below are built with static layout algebra; a tile that
    straddles a head boundary at a non-multiple offset would need a per-element
    mod). Rows beyond the M limit are left at 0 — those lanes never reach gmem
    (the D store is bound-checked) but must not fault.
    """

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        # Stronger than assume_stride_divisibility (32 bits): assume 16-byte
        # stride divisibility so gmem loads can vectorize to 128 bits. The host
        # asserts head_dim (== the row stride of a contiguous table) allows it.
        tensor = getattr(args, self.name)
        divby = 128 // tensor.element_type.width
        new_stride = tuple(
            cute.assume(s, divby=divby) if not cute.is_static(s) else s for s in tensor.stride
        )
        return {
            self.name: cute.make_tensor(
                tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride)
            )
        }

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        head_dim = const_expr(param.shape[1])
        tile_M, tile_N = ctx.tile_M, ctx.tile_N
        assert tile_N % head_dim == 0 or head_dim % tile_N == 0, (
            "rotary epilogue requires tile_N to be a multiple or a divisor of head_dim"
        )
        m_tile = ctx.tile_coord_mnkl[0]
        mCS = param
        if const_expr(ctx.varlen_manager.varlen_m):
            mCS = cute.domain_offset(
                (ctx.varlen_manager.params.cu_seqlens_m[ctx.batch_idx], 0), mCS
            )
        if const_expr(tile_N % head_dim == 0):
            # One tile covers >= 1 whole head: repeat the head_dim columns via a
            # stride-0 mode. The view is identical for every N tile coordinate.
            gCS_rows = cute.local_tile(mCS, (tile_M, head_dim), (m_tile, 0))
            gCS = cute.make_tensor(
                gCS_rows.iterator,
                cute.make_layout(
                    (tile_M, (head_dim, tile_N // head_dim)),
                    stride=(gCS_rows.stride[0], (gCS_rows.stride[1], 0)),
                ),
            )
        else:
            # A head spans several tiles: slice the head's columns for this tile.
            gCS = cute.local_tile(
                mCS,
                (tile_M, tile_N),
                (m_tile, ctx.tile_coord_mnkl[1] % (head_dim // tile_N)),
            )
        tDgCS = ctx.partition_for_epilogue_fn(gCS)
        tDcCS = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((tile_M, tile_N)))
        if const_expr(ctx.tiled_copy_t2r is not None):
            tDgCS = ctx.tiled_copy_r2s.retile(tDgCS)
            tDcCS = ctx.tiled_copy_r2s.retile(tDcCS)
        tDgCS = cute.group_modes(tDgCS, 3, cute.rank(tDgCS))
        tDcCS = cute.group_modes(tDcCS, 3, cute.rank(tDcCS))
        limit_m = min(ctx.varlen_manager.len_m(ctx.batch_idx) - m_tile * tile_M, tile_M)
        full_tile = Boolean(limit_m >= tile_M)
        # Subtile iteration count decides the prefetch depth below.
        num_subtiles_static = const_expr(
            cute.size(
                cute.zipped_divide(
                    cute.make_layout(gemm.cta_tile_shape_mnk[:2]), ctx.epi_tile
                ).shape[1]
            )
        )
        # Two per-subtile fragments (double buffer): subtile i+1's loads are
        # issued in begin_loop(i) so their latency hides behind subtile i's
        # rotation + store. With a single subtile per tile (QK-norm's wide epi
        # tiles), skip the second buffer — it would only add register pressure.
        # Layout-matched to the gmem partition; integer indexing is
        # coordinate-order, so they still line up elementwise with tRS_rD.
        # Rows beyond limit_m keep the initial 0.
        buf0 = cute.make_rmem_tensor_like(tDgCS[None, None, None, 0])
        buf0.fill(0.0)
        if const_expr(num_subtiles_static > 1):
            buf1 = cute.make_rmem_tensor_like(tDgCS[None, None, None, 0])
            buf1.fill(0.0)
        else:
            buf1 = buf0
        bufs = [buf0, buf1]
        # Explicit vector width for the gmem->rmem copy: autovec is too
        # conservative for gmem sources, so we build the copy atom ourselves.
        src0_f = cute.coalesce(cute.filter_zeros(tDgCS[None, None, None, 0]))
        dst0_f = cute.coalesce(cute.filter_zeros(bufs[0]))
        copy_vec = const_expr(
            min(cute.max_common_vector(src0_f, dst0_f), 128 // param.element_type.width)
        )
        assert head_dim % copy_vec == 0, "head_dim must be a multiple of the copy vector"
        # Subtile iteration order, to know which epi_coord comes next. Must
        # match the epi_tile_layout built in GemmBase.epilogue.
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), ctx.epi_tile
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape, order=(0, 1) if const_expr(gemm.epi_m_major) else (1, 0)
        )
        if const_expr(param.element_type != gemm.acc_dtype):
            tDrCS_cvt = cute.make_rmem_tensor(bufs[0].shape, gemm.acc_dtype)
        else:
            tDrCS_cvt = None
        state = [tDgCS, tDcCS, full_tile, limit_m, bufs, tDrCS_cvt, copy_vec, epi_tile_layout]
        # Preload subtile 0 so begin_loop(0) finds it ready.
        self._load_subtile(state, epi_tile_layout.get_hier_coord(0), 0)
        return state

    @cute.jit
    def _load_subtile(self, state, epi_coord, buf_idx: cutlass.Constexpr[int]):
        tDgCS, tDcCS, full_tile, limit_m, bufs, _, copy_vec, _ = state
        tDgCS_cur = tDgCS[None, None, None, epi_coord]
        buf = bufs[buf_idx]
        if full_tile:
            tiler = cute.make_layout(copy_vec)
            copy_atom = copy_utils.get_copy_atom(buf.element_type, copy_vec)
            cute.copy(
                copy_atom,
                cute.zipped_divide(cute.coalesce(cute.filter_zeros(tDgCS_cur)), tiler),
                cute.zipped_divide(cute.coalesce(cute.filter_zeros(buf)), tiler),
            )
        else:
            # Ragged last M tile: per-element row-predicated loads. Slow but
            # only ever runs on the boundary tile.
            tDcCS_cur = tDcCS[None, None, None, epi_coord]
            for i in cutlass.range(cute.size(buf), unroll_full=True):
                if tDcCS_cur[i][0] < limit_m:
                    buf[i] = tDgCS_cur[i]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        bufs, tDrCS_cvt, epi_tile_layout = state[4], state[5], state[7]
        idx = const_expr(cute.crd2idx(epi_coord, epi_tile_layout))
        num_subtiles = const_expr(cute.size(epi_tile_layout))
        if const_expr(idx + 1 < num_subtiles):
            self._load_subtile(state, epi_tile_layout.get_hier_coord(idx + 1), (idx + 1) % 2)
        cur = bufs[idx % 2]
        if const_expr(tDrCS_cvt is not None):
            tDrCS_cvt.store(cur.load().to(tDrCS_cvt.element_type))
            return tDrCS_cvt
        return cur


class RotaryCosSinLoadHost(RotaryCosSinLoad):
    """The epirope op + the generic host-layer schema hooks it predates."""

    fn_port = "value"

    def host_arg_key(self, value):
        from quack.cute_dsl_utils import torch2cute_dtype_map

        # head_dim must be static in the fake (begin() does layout algebra on it).
        return (torch2cute_dtype_map[value.dtype], value.shape[-1])

    def host_fake_arg(self, key, fctx):
        from quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(
            dtype, (cute.sym_int(), head_dim), leading_dim=1, divisibility=128 // dtype.width
        )


class RotaryCosSinTMALoad(TileLoad):
    """The (seqlen_ro, head_dim) interleaved cos/sin table staged through the
    TMA epilogue load pipeline instead of per-tile gmem->rmem copies
    (RotaryCosSinLoad). Default choice — see ``rotary_cos_sin_load``.

    TMA descriptors cannot encode the stride-0 head broadcast, so the wrap
    lives in the copy COORDINATES: each epi subtile TMA-loads the table box
    at column ``output_col % head_dim`` (redundant loads across heads hit L2).
    The smem stage then holds exactly the subtile-aligned cos/sin slice, so
    the whole TileLoad consumer path (S2R, staging, pipeline tx accounting)
    is inherited unchanged. Dense-only for now (TMA loads have no varlen_m
    ragged-descriptor path; use the LDG op under varlen).
    """

    def __init__(self, name):
        super().__init__(name)

    def host_arg_key(self, value):
        from quack.cute_dsl_utils import torch2cute_dtype_map

        # head_dim static in the fake: the copy fn branches on it at trace time.
        return (torch2cute_dtype_map[value.dtype], value.shape[-1])

    def host_fake_arg(self, key, fctx):
        from quack.compile_utils import make_fake_tensor

        dtype, head_dim = key
        return make_fake_tensor(
            dtype, (cute.sym_int(), head_dim), leading_dim=1, divisibility=128 // dtype.width
        )

    def load_g2s_copy_fn(
        self,
        gemm,
        params,
        smem_tensor,
        tile_coord_mnkl,
        varlen_manager,
        epi_pipeline,
    ):
        assert not varlen_manager.varlen_m, (
            "RotaryCosSinTMALoad does not support varlen_m; use "
            "rotary_cos_sin_load(name, tma=False) (rope_table_ldg_epi / qk_rope_ldg_epi)"
        )
        # The TMA prep appends a stride-0 batch mode; slice it off (all
        # batches share the table).
        tensor = varlen_manager.offset_batch_epi(getattr(params, self.name), tile_coord_mnkl[3])
        head_dim = const_expr(tensor.shape[1])
        tile_M, tile_N = gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[1]
        epi_tile = getattr(params, self._epi_tile_key())
        epi_N = const_expr(cute.size(epi_tile[1]))
        atom = getattr(params, self._tma_atom_key())
        if const_expr(tile_N % head_dim == 0):
            # Tile covers >= 1 whole head: tile the table (tile_M, head_dim)
            # and wrap the epi-subtile N coordinate modulo subtiles-per-head.
            assert head_dim % epi_N == 0, "head_dim must be a multiple of the epi tile N"
            subtiles_per_head = const_expr(head_dim // epi_N)
            copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
                atom,
                tensor,
                (tile_M, head_dim),
                epi_tile,
                smem_tensor,
                (tile_coord_mnkl[0], 0),
            )
            inner = copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)
            # Callers pass either the hier (epi_m, epi_n) coord (gemm_base's
            # inline path) or a linear subtile index (the SM100 dedicated
            # epi-load warp): decode linear via the same ordered layout the
            # store loop uses so pipeline stage order matches consumption.
            epi_tile_shape = cute.zipped_divide(
                cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
            ).shape[1]
            epi_tile_layout = cute.make_ordered_layout(
                epi_tile_shape, order=(0, 1) if const_expr(gemm.epi_m_major) else (1, 0)
            )

            def copy_fn(src_idx, producer_state, **kw):
                coord = (
                    src_idx
                    if isinstance(src_idx, tuple)
                    else epi_tile_layout.get_hier_coord(src_idx)
                )
                inner((coord[0], coord[1] % subtiles_per_head), producer_state, **kw)

            return copy_fn
        # A head spans several tiles: pick the head-relative N tile, no
        # per-subtile wrap needed.
        assert head_dim % tile_N == 0, (
            "rotary epilogue requires tile_N to be a multiple or a divisor of head_dim"
        )
        copy_tile_fn, _, _ = gemm.epilog_gmem_copy_and_partition(
            atom,
            tensor,
            (tile_M, tile_N),
            epi_tile,
            smem_tensor,
            (tile_coord_mnkl[0], tile_coord_mnkl[1] % (head_dim // tile_N)),
        )
        return copy_utils.tma_producer_copy_fn(copy_tile_fn, epi_pipeline)


def rotary_cos_sin_load(name, tma=True):
    """Rotary cos/sin table op. ``tma=True`` (default) stages the table via
    the TMA epilogue load pipeline; ``tma=False`` is the per-tile gmem->rmem
    op — required for varlen_m and pre-TMA archs (SM80), and 1-3% faster at
    large non-pingpong tiles.

    Why TMA is the default — H100, m=16384 k=4096 head_dim=128, rope overhead
    vs an identity epilogue at the same config (interleaved-median bench):

    ==================  =======  =======
    config              LDG      TMA
    ==================  =======  =======
    256x128 c(1,2)      +15.3%   +15.5%   best PLAIN-GEMM config
    128x256 c(1,1)      +13.6%   +10.0%
    128x128 c(1,2) pp    +6.2%    +1.7%
    192x128 c(1,2) pp    +4.3%    +1.3%   best rope/qknorm config
    ==================  =======  =======

    Under pingpong the producer-warp TMA staging escapes the exclusive
    per-warpgroup epilogue window that serializes consumer LDGs; at 192-row
    pingpong tiles the LDG double-buffer register cost tips composed
    epilogues into spills (qk_rope: LDG +23% vs TMA +2.5%). Clustered
    pingpong + TMA is the absolute winner for the fused QK-norm/RoPE
    epilogues (qk_rope 794us vs 918us at the best non-pingpong config), so
    the default optimizes the configs these epilogues actually run on.
    """
    return RotaryCosSinTMALoad(name) if tma else RotaryCosSinLoadHost(name)


@gemm_epilogue(ops={"cs": rotary_cos_sin_load("cs")}, mode="acc_pair")
def rope_table_epi(acc, cs, bias):
    """RoPE from the real (seqlen, head_dim) table op, composed with a rowvec
    bias in fn math — the op is a value port; rotation order is explicit."""
    x1, x2 = unpack(acc + bias)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


@gemm_epilogue(ops={"cs": rotary_cos_sin_load("cs", tma=False)}, mode="acc_pair")
def rope_table_ldg_epi(acc, cs, bias):
    """rope_table_epi on the gmem->rmem table op: required for varlen_m
    (table indexed by global flattened row), pre-TMA archs, and marginally
    faster at large non-pingpong tiles (see rotary_cos_sin_load)."""
    x1, x2 = unpack(acc + bias)
    c, s = unpack(cs)
    return {"D": pack(x1 * c - x2 * s, x1 * s + x2 * c)}


def make_interleaved_cos_sin(cos, sin):
    """Interleave HF-style cos/sin (seqlen_ro, head_dim/2) into the
    (seqlen_ro, head_dim) table RotaryCosSinLoad consumes (cos at even
    columns, sin at odd). From the epirope hand-written kernel."""
    import torch

    assert cos.shape == sin.shape and cos.ndim == 2
    return torch.stack([cos, sin], dim=-1).reshape(cos.shape[0], -1).contiguous()
