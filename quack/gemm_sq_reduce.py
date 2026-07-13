# Copyright (c) 2025-2026, Tri Dao.
# GEMM with column vector reduction of squared output and optional rowvec scaling:
# D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.

from typing import NamedTuple, Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from quack.cute_dsl_utils import mlir_namedtuple, get_device_capacity
from quack.epi_ops import (
    ColVecReduce,
    RowVecLoad,
    Scalar,
    TileStore,
    colvec_reduce_accumulate,
    vec_multiply,
)
from quack.gemm_act import GemmActMixin
from quack.gemm_host import (
    GemmEpiPlan,
    build_gemm_epi_plan,
    gemm_epi_plan_key,
    run_gemm_epi_plan,
)
from quack.gemm_sm80 import GemmSm80
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
from quack.gemm_sm120 import GemmSm120
from quack.rounding import RoundingMode
import quack.utils as utils


class GemmSqReduceMixin(GemmActMixin):
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    The sq_sum is computed BEFORE the rowvec scaling. If mAuxOut is provided, the
    pre-rowvec value (D_raw, after alpha/beta/C) is written to it.
    """

    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        Scalar("sr_seed", dtype=Int32),
        RowVecLoad("mRowVecBroadcast"),
        ColVecReduce("mColVecReduce"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = ()  # no act_fn

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecReduce: Optional[cute.Tensor] = None
        mAuxOut: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None

    # EpilogueParams auto-generated from _epi_ops

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        if args.mAuxOut is not None:
            self.aux_out_dtype = args.mAuxOut.element_type
            self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
            self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        d = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**d)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrColVecReduce = epi_loop_tensors.get("mColVecReduce")
        tDrRowVec = epi_loop_tensors.get("mRowVecBroadcast")
        # Load accumulator, apply alpha/beta/C (skip rowvec/colvec — we handle rowvec below)
        rD = tRS_rD.load()
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        # Accumulate sq_sum BEFORE rowvec scaling: reduce[m] += sum_n(D[m,n]^2)
        colvec_reduce_accumulate(self, tDrColVecReduce, tRS_rD, rScale=tRS_rD)
        # Snapshot pre-rowvec value if the caller wants the aux output written.
        if const_expr(getattr(params, "mAuxOut", None) is not None):
            tRS_rAuxOut = cute.make_rmem_tensor_like(tRS_rD)
            tRS_rAuxOut.store(tRS_rD.load())
            tRS_rAuxOuts = (tRS_rAuxOut,)
        else:
            tRS_rAuxOuts = ()
        # Multiply by rowvec (norm_weight) AFTER sq_sum
        vec_multiply(self, tRS_rD, None, tDrRowVec)
        return tRS_rAuxOuts


class GemmSqReduceSm90(GemmSqReduceMixin, GemmSm90):
    pass


class GemmSqReduceSm80(GemmSqReduceMixin, GemmSm80):
    pass


class GemmSqReduceSm100(GemmSqReduceMixin, GemmSm100):
    pass


class GemmSqReduceSm120(GemmSqReduceMixin, GemmSm120):
    pass


_gemm_sq_reduce_sm_to_cls = {
    8: GemmSqReduceSm80,
    9: GemmSqReduceSm90,
    10: GemmSqReduceSm100,
    11: GemmSqReduceSm100,
    12: GemmSqReduceSm120,
}

_gemm_sq_reduce_plan_cache: dict[tuple, GemmEpiPlan] = {}


def gemm_sq_reduce(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Tensor,  # (l, m, n)
    C: Optional[Tensor],  # (l, m, n)
    colvec_reduce: Tensor,  # (l, m, ceildiv(n, tile_n))
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    tile_K: int | None = None,
    pingpong: bool = False,
    persistent: bool = True,
    is_dynamic_persistent: bool = False,
    max_swizzle_size: int = 8,
    rowvec: Optional[Tensor] = None,  # (l, n) — norm_weight
    aux_out: Optional[Tensor] = None,  # (l, m, n) — pre-rowvec output snapshot
    b_kn: bool = False,  # B passed (k, n) / (l, k, n), transposed at trace time (SM90+)
) -> GemmEpiPlan:
    """GEMM + sq_reduce + optional rowvec scaling.

    D_raw = A @ B (+ C), colvec_reduce[m] = sum_n(D_raw[m,n]^2), D_out = D_raw * rowvec.
    If aux_out is provided, the pre-rowvec value (D_raw, after alpha/beta/C) is also
    written to it.
    """
    epi_values = dict(mRowVecBroadcast=rowvec, mColVecReduce=colvec_reduce, mAuxOut=aux_out)
    key = gemm_epi_plan_key(
        A,
        B,
        D,
        C,
        epi_values,
        None,
        tile_count_semaphore is not None,
        A.device,
        tile_M,
        tile_N,
        tile_K,
        cluster_M,
        cluster_N,
        pingpong,
        persistent,
        is_dynamic_persistent,
        max_swizzle_size,
        b_kn,
    )
    plan = _gemm_sq_reduce_plan_cache.get(key)
    if plan is None:
        plan = _build_gemm_sq_reduce_plan(
            A,
            B,
            D,
            C,
            colvec_reduce,
            tile_count_semaphore=tile_count_semaphore,
            tile_M=tile_M,
            tile_N=tile_N,
            cluster_M=cluster_M,
            cluster_N=cluster_N,
            tile_K=tile_K,
            pingpong=pingpong,
            persistent=persistent,
            is_dynamic_persistent=is_dynamic_persistent,
            max_swizzle_size=max_swizzle_size,
            rowvec=rowvec,
            aux_out=aux_out,
            b_kn=b_kn,
        )
        _gemm_sq_reduce_plan_cache[key] = plan
    run_gemm_sq_reduce_plan(
        plan,
        A,
        B,
        D,
        C,
        colvec_reduce,
        tile_count_semaphore=tile_count_semaphore,
        rowvec=rowvec,
        aux_out=aux_out,
    )
    return plan


def run_gemm_sq_reduce_plan(
    plan: GemmEpiPlan,
    A: Tensor,
    B: Tensor,
    D: Tensor,
    C: Optional[Tensor],
    colvec_reduce: Tensor,
    *,
    tile_count_semaphore: Optional[Tensor] = None,
    rowvec: Optional[Tensor] = None,
    aux_out: Optional[Tensor] = None,
) -> None:
    """Launch a resolved plan: only per-call pointers here.
    The tensors must match the metadata the plan was built from."""
    run_gemm_epi_plan(
        plan,
        A,
        B,
        D,
        C,
        dict(mRowVecBroadcast=rowvec, mColVecReduce=colvec_reduce, mAuxOut=aux_out),
        tile_count_semaphore=tile_count_semaphore,
    )


def _build_gemm_sq_reduce_plan(
    A,
    B,
    D,
    C,
    colvec_reduce,
    *,
    tile_count_semaphore,
    tile_M,
    tile_N,
    cluster_M,
    cluster_N,
    tile_K,
    pingpong,
    persistent,
    is_dynamic_persistent,
    max_swizzle_size,
    rowvec,
    aux_out,
    b_kn=False,
) -> GemmEpiPlan:
    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [8, 9, 10, 11, 12], (
        "Only SM8x, SM90, SM100, SM110, and SM120 are supported"
    )
    batched = A.ndim == 3
    if not batched:
        # Dense 2D (unbatched) operands: trace-time batch append, see gemm_act.
        # colvec_reduce must then be 2D (m, n_tiles) — its ndim keys the fake.
        assert (
            B.ndim == 2
            and D.ndim == 2
            and (C is None or C.ndim == 2)
            and colvec_reduce.ndim == 2
            and (aux_out is None or aux_out.ndim == 2)
        ), "2D (unbatched) A requires 2D B, D, C, colvec_reduce, and aux_out"
    if not batched or b_kn:
        assert device_capacity[0] in [9, 10, 11, 12], "2D (unbatched) operands / b_kn require SM90+"

    if is_dynamic_persistent and device_capacity[0] == 9:
        assert tile_count_semaphore is not None, (
            "Dynamic persistent tile scheduler in SM90 requires a semaphore in GMEM"
        )

    return build_gemm_epi_plan(
        _gemm_sq_reduce_sm_to_cls[device_capacity[0]],
        device_capacity,
        A,
        B,
        D,
        C,
        epi_values=dict(mRowVecBroadcast=rowvec, mColVecReduce=colvec_reduce, mAuxOut=aux_out),
        tile_M=tile_M,
        tile_N=tile_N,
        cluster_M=cluster_M,
        cluster_N=cluster_N,
        tile_K=tile_K,
        pingpong=pingpong,
        persistent=persistent,
        is_dynamic_persistent=is_dynamic_persistent,
        max_swizzle_size=max_swizzle_size,
        b_kn=b_kn,
    )
