# Copyright (c) 2025, Tri Dao.
from typing import Optional

from torch import Tensor

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
import cutlass.torch as cutlass_torch

from quack.gemm_act_sm90 import GemmActSm90
from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm_wrapper_utils import GemmWrapperBase
import quack.activation


class GemmDActSm90(GemmActSm90):
    # Different from GemmActSm90, here act_bwd_fn must take in 2 arguments (x, dout)
    # and return 2 arguments (dx, out)
    EpilogueArguments = GemmActSm90.EpilogueArguments
    EpilogueParams = GemmActSm90.EpilogueParams

    @cute.jit
    def epi_visit_acc_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        assert tRS_rC is not None
        tRS_rC_acc = cute.make_fragment_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_fragment(tRS_rD.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rD[i], tRS_rPostAct[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
        else:
            tRS_rPostAct = tRS_rC_acc
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


dact_fn_map = {
    None: None,
    "relu": quack.activation.drelu,
    "relu_sq": quack.activation.drelu_sq,
    "gelu_tanh_approx": quack.activation.dgelu_tanh_approx,
}


def gemm_dact_sm90(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    PreAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
) -> None:
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert Out.stride(-1) == 1, "varlen_m requires Out to be n-major"
        assert PreAct.stride(-1) == 1, "varlen_m requires PreAct to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    assert activation in dact_fn_map, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, Out, PreAct, additional_tensors={"PostAct": PostAct}, cu_seqlens_m=cu_seqlens_m
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

    acc_dtype = cutlass.Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmDActSm90.is_valid_dtypes(
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
    act_fn = dact_fn_map[activation]
    epi_args = GemmDActSm90.EpilogueArguments(tensor_infos["PostAct"].cute_tensor, act_fn)
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmDActSm90.num_epi_tensormaps,
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
        cu_seqlens_m is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_dact_sm90.compile_cache
    if compile_key not in cache:
        gemm = GemmDActSm90(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            pingpong=pingpong,
            is_persistent=persistent,
        )
        cache[compile_key] = cute.compile(
            gemm,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            None,  # mAIdx
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
        None,
        current_stream,
    )


gemm_dact_sm90.compile_cache = {}
