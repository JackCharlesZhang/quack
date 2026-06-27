# Copyright (c) 2026, Tri Dao.
"""Small CuTe tensor convenience helpers.

Importing this module intentionally mutates CuTe's tensor class process-wide so
fragments can be written in a more PyTorch-like style::

    rmem_f32 = rmem_f16.to(Float32)
    rmem_contig = rmem_view.contiguous()

``tensor.to(dtype)`` is exactly the explicit CuTe sequence::

    dst = cute.make_rmem_tensor_like(src, dtype)
    dst.store(src.load().to(dtype))

``tensor.contiguous()`` mirrors ``quack.copy_utils.contiguous``.
"""

from __future__ import annotations

from typing import Any

import cutlass.cute as cute
import cutlass.cute.tensor as _cute_tensor
from cutlass.cutlass_dsl import dsl_user_op


_ORIGINAL_TO_ATTR = "_quack_original_to"
_PATCHED_TO_ATTR = "_quack_rmem_tensor_to"
_ORIGINAL_CONTIGUOUS_ATTR = "_quack_original_contiguous"
_PATCHED_CONTIGUOUS_ATTR = "_quack_tensor_contiguous"


def _make_to() -> Any:
    @dsl_user_op
    def _to(self: Any, dtype: Any, *, loc: Any = None, ip: Any = None) -> Any:
        if self.memspace != cute.AddressSpace.rmem:
            raise ValueError("Tensor.to(dtype) is only supported for rmem tensors")

        dst = cute.make_rmem_tensor_like(self, dtype, loc=loc, ip=ip)
        dst.store(self.load(loc=loc, ip=ip).to(dtype, loc=loc, ip=ip), loc=loc, ip=ip)
        return dst

    return _to


def _make_contiguous() -> Any:
    @dsl_user_op
    def _contiguous(self: Any, *, loc: Any = None, ip: Any = None) -> Any:
        dst = cute.make_rmem_tensor(self.shape, self.element_type, loc=loc, ip=ip)
        cute.autovec_copy(self, dst, loc=loc, ip=ip)
        return dst

    return _contiguous


def patch_cute_tensor() -> None:
    """Monkey patch CuTe tensors with QuACK convenience methods.

    The patch is idempotent. CuTe's immutable ``TensorSSA.to`` already handles
    value conversion; this installs the analogous materializing conversion on
    mutable register-backed ``_Tensor`` fragments, plus a ``contiguous`` method
    equivalent to :func:`quack.copy_utils.contiguous`.
    """
    tensor_cls = _cute_tensor._Tensor
    if _PATCHED_TO_ATTR not in tensor_cls.__dict__:
        original_to = getattr(tensor_cls, "to", None)
        if original_to is not None:
            setattr(tensor_cls, _ORIGINAL_TO_ATTR, original_to)
        tensor_cls.to = _make_to()  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_TO_ATTR, True)

    if _PATCHED_CONTIGUOUS_ATTR not in tensor_cls.__dict__:
        original_contiguous = getattr(tensor_cls, "contiguous", None)
        if original_contiguous is not None:
            setattr(tensor_cls, _ORIGINAL_CONTIGUOUS_ATTR, original_contiguous)
        tensor_cls.contiguous = _make_contiguous()  # type: ignore[method-assign]
        setattr(tensor_cls, _PATCHED_CONTIGUOUS_ATTR, True)


patch_cute_tensor()


__all__ = ["patch_cute_tensor"]
