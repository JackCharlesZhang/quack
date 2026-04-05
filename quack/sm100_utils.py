# Copyright (c) 2025, Tri Dao.

from typing import Type, Union, Literal

import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_og
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cutlass_dsl import Numeric, dsl_user_op


@dsl_user_op
def make_smem_layout_cpasync_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """
    :param tiled_mma: The tiled MMA used to partition tensor A
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The MMA tile shape
    :type mma_tiler_mnk: cute.cute.Tile
    :param a_dtype: The element type for tensor A
    :type a_dtype: Type[Numeric]
    :param num_stages: The number of pipeline stages for tensor A
    :type num_stages: int

    :return: SMEM layout for tensor A
    :rtype: Union[cute.Layout, cute.ComposedLayout]
    """

    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode,
            a_dtype,
            a_smem_shape_mn_k,
            loc=loc,
            ip=ip,
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=((1, 0, 2) if not is_k_major else (0, 1, 2)),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged


@dsl_user_op
def make_smem_layout_tma_gather_a(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    a_dtype: Type[Numeric],
    num_stages: int,
    gather_dim: Literal["m", "k"],
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """SMEM load layout for A with TMA gather4.

    gather_dim="m": gather in M dimension (K-major A, varlen_m).
      Replaces M-atom with 4, keeps (M, K) order.
    gather_dim="k": gather in K dimension (M-major A, varlen_k).
      Transposes atom so K-groups of 4 lead, for partition_S alignment.
    """
    is_k_major = tiled_mma.op.a_major_mode == OperandMajorMode.K
    if gather_dim == "m":
        assert is_k_major, "TMA gather M requires K-major layout for tensor A"
    else:
        assert not is_k_major, "TMA gather K requires M-major layout for tensor A"
    a_smem_shape = tiled_mma.partition_shape_A(
        cute.dice(mma_tiler_mnk, (1, None, 1), loc=loc, ip=ip)
    )
    a_smem_shape_mn_k = (
        cute.size(a_smem_shape[0][0], loc=loc, ip=ip) * a_smem_shape[1],
        cute.size(a_smem_shape[0][1], loc=loc, ip=ip) * a_smem_shape[2],
    )
    # e,g., S<3, 4, 3> o  0 o (8, 64):(64, 1) for k_major
    # e,g., S<3, 4, 3> o  0 o (64, 8):(1, 64) for m_major
    a_smem_layout_atom = sm100_utils_og.make_smem_layout_atom(
        sm100_utils_og.get_smem_layout_atom_ab(
            tiled_mma.op.a_major_mode, a_dtype, a_smem_shape_mn_k, loc=loc, ip=ip
        ),
        a_dtype,
        loc=loc,
        ip=ip,
    )
    swizzle = a_smem_layout_atom.inner
    smem_layout = a_smem_layout_atom.outer
    if gather_dim == "m":
        # Replace M-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout((4, smem_layout.shape[1]), stride=smem_layout.stride, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    else:
        # Replace K-dim with 4 for gather4, keep original strides
        a_smem_layout_atom = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout((smem_layout.shape[1], 4), stride=smem_layout.stride, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape_mn_k, num_stages, loc=loc, ip=ip),
        order=(1, 0, 2) if not is_k_major else (0, 1, 2),
        loc=loc,
        ip=ip,
    )
    return a_smem_layout_staged
