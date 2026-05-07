# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""``cute_op``: ``torch.library.custom_op`` for CuTe DSL kernels.

Same trick as ``torch.library.triton_op`` (register the impl as the fake/meta
kernel too), specialized for our setup:

* Under ``torch.compile`` we stay a complete no-op (matches prior behavior;
  also avoids moving compile latency into dynamo trace time).
* Under ``FakeTensorMode`` with SymInt shapes (dynamic-shape tracing), skip:
  ``@jit_cache`` is an ``lru_cache`` and SymInts are unhashable.
* Otherwise (``FakeTensorMode`` with concrete shapes, e.g. the COMPILE_ONLY
  worker) flip ``cache_utils.COMPILE_ONLY`` for the duration of the call so
  ``@jit_cache`` returns ``_noop_kernel`` for every ``_compile_*(...)`` it
  populates. The body runs end-to-end, the .o cache is filled, and no kernel
  is actually launched.

This removes the need for hand-written ``_*_fake`` twins on each op.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

import torch

from quack import cache_utils


__all__ = ["cute_op"]


def _has_symint_shape(args: Iterable[Any]) -> bool:
    for a in args:
        if isinstance(a, torch.Tensor) and any(isinstance(s, torch.SymInt) for s in a.shape):
            return True
    return False


def cute_op(
    name: str,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
    device_types: Optional[Union[str, Iterable[str]]] = None,
) -> Callable:
    """Like ``torch.library.triton_op``, but for CuTe DSL kernels.

    Args:
        name: ``"namespace::op_name"``.
        mutates_args: Names of mutated tensor args.
        schema: Optional explicit schema. Required when mutating an
            ``Optional[Tensor]`` arg (PyTorch can't infer those).
        device_types: Optional device-type restriction.
    """

    def dec(fn: Callable) -> Any:
        kwargs: dict[str, Any] = {"mutates_args": mutates_args}
        if schema is not None:
            kwargs["schema"] = schema
        if device_types is not None:
            kwargs["device_types"] = device_types
        op = torch.library.custom_op(name, fn, **kwargs)

        @op.register_fake
        def _fake(*args, **kw):
            if torch.compiler.is_compiling():
                return
            if _has_symint_shape(args) or _has_symint_shape(kw.values()):
                return
            saved = cache_utils.COMPILE_ONLY
            cache_utils.COMPILE_ONLY = True
            try:
                fn(*args, **kw)
            finally:
                cache_utils.COMPILE_ONLY = saved

        return op

    return dec
