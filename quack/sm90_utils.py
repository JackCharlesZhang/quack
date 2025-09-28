# Copyright (c) 2025, Tri Dao.

from typing import Type, Union

import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.utils import LayoutEnum


@dsl_user_op
def make_smem_layout_epi(
    epi_dtype: Type[Numeric],
    epi_layout: LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    epilog_shape = cute.product_each(cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip)
    epi_major_mode_size = epilog_shape[1] if epi_layout.is_n_major_c() else epilog_shape[0]
    epi_smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(epi_layout, epi_dtype, epi_major_mode_size),
        epi_dtype,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        epi_smem_layout_atom,
        cute.append(epilog_shape, epi_stage),
        order=(1, 0, 2) if epi_layout.is_m_major_c() else (0, 1, 2),
    )
    return epi_smem_layout_staged
