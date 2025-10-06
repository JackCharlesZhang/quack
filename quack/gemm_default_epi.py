# Copyright (c) 2025, Wentao Guo, Tri Dao.
from typing import Optional
from dataclasses import dataclass


import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack.cute_dsl_utils import ArgumentsBase, ParamsBase
from quack.gemm_sm90 import GemmSm90
from quack.gemm_sm100 import GemmSm100
import quack.utils as utils


class GemmDefaultEpiMixin:
    num_epi_tensormaps: int = 0

    @dataclass
    class EpilogueArguments(ArgumentsBase):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        add_to_output: bool = False

    @dataclass
    class EpilogueParams(ParamsBase):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        return self.EpilogueParams(alpha=args.alpha, beta=args.beta)

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        rD = tRS_rD.load()
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        return None


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    pass


class GemmDefaultSm100(GemmDefaultEpiMixin, GemmSm100):
    pass
