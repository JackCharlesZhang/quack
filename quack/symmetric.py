from typing import Optional
from functools import partial
from torch import Tensor
from quack.gemm_act import GemmActMixin, GemmActSm90, act_fn_map, gemm_act
from quack.gemm_sm90 import GemmSm90
from quack.tile_scheduler import TriangularTileScheduler
from quack.gemm_wrapper_utils import GemmWrapperBase
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Float32
from cutlass.cute.runtime import make_ptr

class GemmSymmetric(GemmActMixin, GemmSm90):
    def get_scheduler_class(self, varlen_m: bool = False):
        return TriangularTileScheduler

def gemm_symmetric(
    A: Tensor,  # (l, m, k) or (total_m, k) if varlen_m or (whatever, k) if gather_A with varlen_m
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    PostAct: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) if gather_A with varlen_m
) -> None:
    if cu_seqlens_m is not None:
        assert persistent, "varlen_m requires persistent=True"
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        if D is not None:
            assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
        assert PostAct.stride(-1) == 1, "varlen_m requires PostAct to be n-major"
    gather_A = A_idx is not None
    if gather_A:
        assert cu_seqlens_m is not None, "gather_A requires varlen (cu_seqlens_m must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    assert activation in act_fn_map, f"Unsupported activation {activation}"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, additional_tensors={"PostAct": PostAct}, cu_seqlens_m=cu_seqlens_m, A_idx=A_idx
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

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] in [9], "Only SM90 is supported"
    GemmCls = GemmActSm90
    # TODO: implement dynamic persistent
    if device_capacity[0] > 9:
        tile_count_semaphore = None

    acc_dtype = cutlass.Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors({k: v for k, v in tensor_infos.items()}, major_configs)

    # tensor_infos["PostAct"].cute_tensor = GemmWrapperBase.create_transposed_cute_tensor(tensor_infos["PostAct"].tensor, tensor_infos["PostAct"].major, major_configs["PostAct"])
    
    
    def scalar_arg(scalar: float | Tensor):
        if isinstance(scalar, float):
            return Float32(scalar) if scalar != 1.0 else None
        else:
            assert isinstance(scalar, Tensor)
            return make_ptr(Float32, scalar.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    act_fn = act_fn_map[activation]
    epi_args = GemmCls.EpilogueArguments(tensor_infos["PostAct"].cute_tensor, act_fn, scalar_arg(alpha), scalar_arg(beta))
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters, tile_count_semaphore, max_swizzle_size=max_swizzle_size
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen_m)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        None,  # cu_seqlens_k
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
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
        device_capacity,
        max_swizzle_size,
        2 if isinstance(alpha, Tensor) else (1 if alpha == 1.0 else 0),
        2 if isinstance(beta, Tensor) else (1 if beta == 1.0 else 0),
        cu_seqlens_m is not None,
        A_idx is not None,
        key_tensor_names=("A", "B", "D", "PostAct", "C"),
    )
    cache = gemm_act.compile_cache
    if compile_key not in cache:
        if device_capacity[0] == 9:
            GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
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
        current_stream,
    )


gemm_act.compile_cache = {}
