# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack, make_ptr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.varlen_utils import VarlenArguments
from quack.dense_gemm_sm90 import TileSchedulerOptions


@dataclass
class GemmTensorInfo:
    tensor: Optional[Tensor]
    dtype: Optional[Any] = None
    major: Optional[str] = None
    cute_tensor: Optional[cute.Tensor] = None


class GemmWrapperBase:
    @staticmethod
    def validate_tensor_3d(tensor: Tensor, name: str) -> None:
        assert tensor.dim() == 3 and tensor.is_cuda, f"{name} must be a 3D CUDA tensor"
        assert tensor.dtype in torch2cute_dtype_map, f"Unsupported dtype for {name}"

    @staticmethod
    def validate_shape(tensor: Tensor, expected_shape: Tuple[int, ...], name: str) -> None:
        assert tensor.shape == expected_shape, (
            f"{name} must have shape {expected_shape}, got {tensor.shape}"
        )

    @staticmethod
    def get_major_order(tensor: Tensor, dims: Tuple[str, str, str]) -> str:
        # Tensor is already permuted to (dims[0], dims[1], dims[2])
        # stride(1) == 1 means dims[1] is contiguous (innermost)
        return dims[1] if tensor.stride(1) == 1 else dims[0]

    @staticmethod
    def create_cute_tensor(
        tensor: Optional[Tensor],
        major: Optional[str],
        dims: Tuple[str, str, str],
        assumed_align: int = 16,
    ) -> Optional[cute.Tensor]:
        if tensor is None:
            return None
        # Tensor is already permuted to (dims[0], dims[1], dims[2]) or (dim[0], dim[1])
        # If major is dims[1], leading_dim is 1; if major is dims[0], leading_dim is 0
        leading_dim = 1 if major == dims[1] else 0
        return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    @staticmethod
    def validate_and_prepare_tensors(
        A: Tensor,
        B: Tensor,
        D: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
        additional_tensors: Optional[Dict[str, Tensor]] = None,
        cu_seqlens_m: Optional[Tensor] = None,
    ) -> Tuple[int, int, int, int, Dict[str, GemmTensorInfo]]:
        if cu_seqlens_m is not None:
            # Handle variable length M case
            # A is (total_m, k), B is (l, n, k), D is (total_m, n), C is (total_m, n)
            assert A.dim() == 2, f"A must be 2D when using varlen_m, got {A.dim()}D"
            total_M, K = A.shape
            assert B.dim() == 3, f"B must be 3D with varlen_m, got {B.dim()}D"
            L, N, K_B = B.shape
            assert K == K_B, f"K dimension mismatch: A has {K}, B has {K_B}"
            assert cu_seqlens_m.shape == (L + 1,), (
                f"cu_seqlens_m must have shape ({L + 1},), got {cu_seqlens_m.shape}"
            )
            assert B.dtype == A.dtype, "A and B must have the same dtype"
            if D is not None:
                assert D.dim() == 2, f"D must be 2D when using varlen_m, got {D.dim()}D"
                assert D.shape == (total_M, N), (
                    f"D shape {D.shape} doesn't match expected ({total_M}, {N})"
                )
            if C is not None:
                assert C.dim() == 2, f"C must be 2D when using varlen_m, got {C.dim()}D"
                assert C.shape == (total_M, N), (
                    f"C shape {C.shape} doesn't match expected ({total_M}, {N})"
                )
            M = total_M  # Return total_M as M dimension
        else:
            # Normal case - all tensors must be 3D
            GemmWrapperBase.validate_tensor_3d(A, "A")
            L, M, K = A.shape
            GemmWrapperBase.validate_tensor_3d(B, "B")
            _, N, _ = B.shape
            assert B.dtype == A.dtype, "A and B must have the same dtype"
            GemmWrapperBase.validate_shape(B, (L, N, K), "B")
            if D is not None:
                GemmWrapperBase.validate_tensor_3d(D, "D")
                GemmWrapperBase.validate_shape(D, (L, M, N), "D")
            if C is not None:
                GemmWrapperBase.validate_tensor_3d(C, "C")
                GemmWrapperBase.validate_shape(C, (L, M, N), "C")

        tensors = {
            "A": GemmTensorInfo(A),
            "B": GemmTensorInfo(B),
            "D": GemmTensorInfo(D),
            "C": GemmTensorInfo(C),
        }

        if additional_tensors:
            for name, tensor in additional_tensors.items():
                if tensor is not None:
                    if cu_seqlens_m is not None:
                        assert tensor.dim() == 2, f"{name} must be 2D when using varlen_m"
                        assert tensor.shape == (total_M, N), f"{name} shape mismatch"
                    else:
                        GemmWrapperBase.validate_tensor_3d(tensor, name)
                        GemmWrapperBase.validate_shape(tensor, (L, M, N), name)
                tensors[name] = GemmTensorInfo(tensor)

        return L, M, K, N, tensors

    @staticmethod
    def permute_tensors(tensors: Dict[str, GemmTensorInfo], varlen_m: bool = False) -> None:
        if varlen_m:
            # In varlen_m case: A, D, C are 2D (already in right shape), B is 3D
            for name, info in tensors.items():
                if info.tensor is not None and name == "B":
                    # Permute B from (L, N, K) -> (N, K, L)
                    info.tensor = info.tensor.permute(1, 2, 0)
        else:
            # Normal case: permute all 3D tensors from (L, M, K/N) to (M, K/N, L)
            for info in tensors.values():
                if info.tensor is not None:
                    info.tensor = info.tensor.permute(1, 2, 0)

    @staticmethod
    def extract_dtypes(tensors: Dict[str, GemmTensorInfo]) -> None:
        for info in tensors.values():
            if info.tensor is not None:
                info.dtype = torch2cute_dtype_map[info.tensor.dtype]

    @staticmethod
    def determine_major_orders(
        tensors: Dict[str, GemmTensorInfo], major_configs: Dict[str, Tuple[str, str, str]]
    ) -> None:
        for name, dims in major_configs.items():
            if name in tensors and tensors[name].tensor is not None:
                tensors[name].major = GemmWrapperBase.get_major_order(tensors[name].tensor, dims)

    @staticmethod
    def create_cute_tensors(
        tensors: Dict[str, GemmTensorInfo], major_configs: Dict[str, Tuple[str, str, str]]
    ) -> None:
        for name, info in tensors.items():
            if info.tensor is not None and name in major_configs:
                info.cute_tensor = GemmWrapperBase.create_cute_tensor(
                    info.tensor, info.major, major_configs[name]
                )

    @staticmethod
    def create_scheduler_args(
        max_active_clusters: int, tile_count_semaphore: Optional[Tensor] = None
    ) -> TileSchedulerOptions:
        return TileSchedulerOptions(
            Int32(max_active_clusters),
            tile_count_semaphore=make_ptr(
                Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
            if tile_count_semaphore is not None
            else None,
        )

    @staticmethod
    def create_varlen_args(
        cu_seqlens_m: Optional[Tensor],
        max_active_clusters: int,
        cluster_shape_mnk: Tuple[int, int, int],
        tensors: Dict[str, GemmTensorInfo],
        num_epi_tensormaps: int = 0,
        pingpong: bool = False,
    ) -> Optional[Any]:
        if cu_seqlens_m is None:
            return None
        # When varlen_m, we assume persistent=True
        # Grid size depends on num_active_clusters and cluster size
        cluster_size = cluster_shape_mnk[0] * cluster_shape_mnk[1]
        num_blocks = max_active_clusters * cluster_size
        # Calculate number of tensormaps needed
        # For varlen_m: need tensormaps for D and epilogue tensors
        num_tensormaps = num_epi_tensormaps * (1 if not pingpong else 2)
        if tensors["D"].tensor is not None:
            num_tensormaps += 1 if not pingpong else 2  # D tensormap
        # Create tensormap buffer (each tensormap is 128 bytes = 16 int64s)
        tensormap_size = 128 // 8  # 16 int64s
        if num_tensormaps > 0:
            tensormaps = torch.empty(
                (num_blocks, num_tensormaps, tensormap_size),
                dtype=torch.int64,
                device=cu_seqlens_m.device,
            )
            tensormaps_cute = from_dlpack(tensormaps, assumed_align=128).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2)
            )
        else:
            tensormaps_cute = None

        return VarlenArguments(
            mCuSeqlensM=from_dlpack(cu_seqlens_m, assumed_align=4).mark_layout_dynamic(
                leading_dim=0
            ),
            mCuSeqlensK=None,  # No variable K for now
            mTensormaps=tensormaps_cute,
        )

    @staticmethod
    def get_compile_key(
        tensors: Dict[str, GemmTensorInfo],
        activation: Optional[str],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool,
        persistent: bool,
        has_semaphore: bool,
        *args,
        key_tensor_names: Tuple[str, ...] = ("A", "B", "D", "C"),
    ) -> Tuple:
        key_parts = []
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].dtype)
        key_parts.append(activation)
        key_parts.extend([tile_shape_mn, cluster_shape_mnk])
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].major)
        key_parts.extend([pingpong, persistent, has_semaphore])
        key_parts.extend(args)
        return tuple(key_parts)
