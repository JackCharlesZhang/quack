# Copyright (c) 2025-2026, Tri Dao.
"""Persistent kernel-cache + compile-only utilities for QuACK.

Public API
----------

Persistent ``.o`` cache:
* :func:`jit_cache` \u2014 decorator that wraps a kernel-compile function with
  in-memory + persistent ``.o`` caching (see :mod:`quack.cache.jit`).
* :data:`COMPILE_ONLY`, :data:`CACHE_ENABLED`, :data:`CACHE_DIR`,
  :data:`EXTRA_SOURCE_DIRS` \u2014 mutable runtime flags (see "Flag semantics"
  below).
* :class:`FileLock`, :func:`get_cache_path`, :class:`CacheInfo` \u2014
  supporting types.

Compile-only cache warming:
* :class:`CompileOnlyFakeTensorMode`, :func:`compile_only_mode`,
  :func:`is_compile_only` \u2014 helpers for populating the cache via
  :class:`~torch._subclasses.fake_tensor.FakeTensorMode` (no GPU memory)
  before the real run. See :mod:`quack.cache.compile_only`.

Flag semantics
--------------

``COMPILE_ONLY`` and the other flags are *mutable runtime state*. They live in
this module (:mod:`quack.cache`) as the single source of truth. Internal
QuACK code (:mod:`quack.cache.jit`, :mod:`quack.cache.compile_only`) reads
them by attribute access on the package, e.g. ``quack.cache.COMPILE_ONLY``,
which always returns the current value.

Recommended user-facing access patterns:

  >>> from quack.cache import is_compile_only, compile_only_mode
  >>> if is_compile_only(): ...               # live read via function call
  >>> with compile_only_mode():                # live mutation, auto-restore
  ...     run_kernels()

If you need to mutate the flag directly (e.g. inside a subprocess worker
that lives entirely in compile-only mode), use attribute assignment on the
package:

  >>> import quack.cache
  >>> quack.cache.COMPILE_ONLY = True

Do **not** mutate the flag via ``from quack.cache import COMPILE_ONLY``
followed by reassignment \u2014 that only rebinds your local namespace and
leaves the canonical flag unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Mutable runtime flags. Source of truth.
#
# CRITICAL ORDERING: these MUST be defined before the ``from quack.cache.jit
# import ...`` block below. ``quack/cache/jit.py`` does ``import quack.cache as
# _state`` at its module top; Python returns the partially-initialized package
# object, and ``_state.COMPILE_ONLY`` lookups inside ``jit_cache``'s wrapper
# (live attribute access) rely on these names already existing at that
# checkpoint. Reordering the imports here, even via an auto-formatter, will
# break the first kernel compile with ``AttributeError: module 'quack.cache'
# has no attribute 'COMPILE_ONLY'``.
#
# The defensive unit tests in ``tests/test_cache.py`` exercise this path
# end-to-end so a reordering bug surfaces immediately.
# ---------------------------------------------------------------------------

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: Optional[str] = os.getenv("QUACK_CACHE_DIR", None)
COMPILE_ONLY: bool = False

#: Downstream projects can append directories here to include their sources
#: in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: List[Path] = []


class CompileOnlyStrictError(RuntimeError):
    """Raised by precompile helpers under ``QUACK_COMPILE_ONLY_STRICT=1``.

    Wraps the underlying exception so the reusable pytest plugin's blanket
    swallow hooks (which exist to ignore expected FakeTensor-incompatibility
    errors *after* the kernel has dispatched) can let strict-mode failures
    surface as real test failures.

    Without this distinction, ``QUACK_COMPILE_ONLY_STRICT=1`` looks like it
    works (the inner ``try/except`` does re-raise) but the surrounding
    pytest plugin would force-pass the test anyway, defeating the strict
    mode entirely.
    """


# ---------------------------------------------------------------------------
# Public API surface. Imported AFTER the flags are defined.
# ---------------------------------------------------------------------------

from quack.cache.jit import (  # noqa: E402
    EXPORT_FUNC_NAME,
    LOCK_TIMEOUT,
    CacheInfo,
    FileLock,
    get_cache_path,
    jit_cache,
)
from quack.cache.compile_only import (  # noqa: E402
    CompileOnlyFakeTensorMode,
    compile_only_mode,
    is_compile_only,
)

# ``__all__`` advertises the *recommended* public API only. The mutable
# runtime flags (``COMPILE_ONLY``, ``CACHE_ENABLED``, ``CACHE_DIR``,
# ``EXTRA_SOURCE_DIRS``) are intentionally *not* listed: ``from quack.cache
# import COMPILE_ONLY`` at module top binds a snapshot of the value at import
# time, and later mutations don't propagate — a footgun. Direct access is still
# supported as ``quack.cache.COMPILE_ONLY`` (live attribute read) but the
# recommended patterns are :func:`is_compile_only` (read) and
# :func:`compile_only_mode` (write, context-managed).
__all__ = [
    # Persistent .o cache.
    "jit_cache",
    "CacheInfo",
    "EXPORT_FUNC_NAME",
    "LOCK_TIMEOUT",
    "FileLock",
    "get_cache_path",
    # Compile-only cache warming.
    "CompileOnlyFakeTensorMode",
    "CompileOnlyStrictError",
    "compile_only_mode",
    "is_compile_only",
]
