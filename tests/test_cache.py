# Copyright (c) 2026, Tri Dao.
"""Unit tests for ``quack.cache`` and its compile-only helpers.

These are deliberately *small and synchronous*: they exercise the package's
mutable-state plumbing without launching real CuTe kernels, so they catch
import-order, mutation-visibility, and exception-propagation regressions
fast.
"""

from __future__ import annotations

import importlib

import pytest
import torch


# ---------------------------------------------------------------------------
# Package layout / public-API surface
# ---------------------------------------------------------------------------


def test_public_api_symbols_resolve():
    """Every name advertised in ``quack.cache.__all__`` must actually exist
    on the package and be importable as ``from quack.cache import X``.

    This is the regression test for the brittle import order in
    ``__init__.py``: if a maintainer reorders flag defs vs. submodule
    imports, the package will fail to initialize and this test fires.
    """
    import quack.cache

    for name in quack.cache.__all__:
        assert hasattr(quack.cache, name), (
            f"quack.cache advertises {name!r} in __all__ but it's missing"
        )


def test_mutable_flags_live_on_package_init():
    """The mutable flags must be attributes of the ``quack.cache`` package
    object itself (not just re-exports from a submodule), so that
    ``quack.cache.COMPILE_ONLY = X`` actually sticks.
    """
    import quack.cache

    for name in ("CACHE_ENABLED", "CACHE_DIR", "COMPILE_ONLY", "EXTRA_SOURCE_DIRS"):
        assert name in vars(quack.cache), (
            f"mutable flag {name!r} should be a direct attribute of quack.cache, not a re-export"
        )


def test_jit_module_sees_live_flags():
    """``quack.cache.jit`` reads flags via ``_state`` (the partially-imported
    package). A regression that breaks this would surface as the disk cache
    silently ignoring ``CACHE_ENABLED=0`` or as ``COMPILE_ONLY`` mutations
    not affecting ``jit_cache``.
    """
    import quack.cache
    import quack.cache.jit as jit_module

    # `_state` should be the package object itself.
    assert jit_module._state is quack.cache


# ---------------------------------------------------------------------------
# compile_only_mode() / is_compile_only() semantics
# ---------------------------------------------------------------------------


def test_compile_only_mode_round_trip():
    import quack.cache

    quack.cache.COMPILE_ONLY = False
    assert not quack.cache.is_compile_only()
    with quack.cache.compile_only_mode():
        assert quack.cache.is_compile_only()
        assert quack.cache.COMPILE_ONLY is True
    assert not quack.cache.is_compile_only()
    assert quack.cache.COMPILE_ONLY is False


def test_compile_only_mode_restores_prior_True():
    """Context manager must restore the prior value, not unconditionally
    reset to False."""
    import quack.cache

    quack.cache.COMPILE_ONLY = True
    try:
        with quack.cache.compile_only_mode():
            assert quack.cache.COMPILE_ONLY is True
        assert quack.cache.COMPILE_ONLY is True  # not clobbered
    finally:
        quack.cache.COMPILE_ONLY = False


def test_compile_only_mode_restores_on_exception():
    import quack.cache

    quack.cache.COMPILE_ONLY = False
    with pytest.raises(RuntimeError, match="boom"):
        with quack.cache.compile_only_mode():
            assert quack.cache.is_compile_only()
            raise RuntimeError("boom")
    assert not quack.cache.is_compile_only()


def test_compile_only_mode_nested():
    import quack.cache

    quack.cache.COMPILE_ONLY = False
    with quack.cache.compile_only_mode():
        with quack.cache.compile_only_mode():
            assert quack.cache.is_compile_only()
        # Outer still active after inner exit (prev=True saved by inner).
        assert quack.cache.is_compile_only()
    assert not quack.cache.is_compile_only()


# ---------------------------------------------------------------------------
# CompileOnlyFakeTensorMode dispatch override
# ---------------------------------------------------------------------------


def test_fake_mode_intercepts_item_int():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.zeros(3, dtype=torch.int32, device="cuda")
        # Stock FakeTensorMode would raise DataDependentOutputException here.
        assert t.sum().item() == _INT_SENTINEL


def test_fake_mode_intercepts_item_float():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _FLOAT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.randn(3, dtype=torch.float32, device="cuda")
        assert t.sum().item() == _FLOAT_SENTINEL


def test_fake_mode_intercepts_tolist():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        # tolist() dispatches aten._local_scalar_dense per element.
        assert t.tolist() == [_INT_SENTINEL] * 4


def test_fake_mode_intercepts_int_cast():
    from quack.cache import CompileOnlyFakeTensorMode
    from quack.cache.compile_only import _INT_SENTINEL

    with CompileOnlyFakeTensorMode():
        t = torch.tensor(5, dtype=torch.int32, device="cuda")
        assert int(t) == _INT_SENTINEL
        # int-dtype FakeTensor → int sentinel → Python casts to float
        assert float(t) == float(_INT_SENTINEL)


def test_fake_mode_passes_through_other_ops():
    """Only ``aten._local_scalar_dense`` is intercepted; everything else
    must keep flowing through ``super().dispatch``."""
    from quack.cache import CompileOnlyFakeTensorMode

    with CompileOnlyFakeTensorMode():
        a = torch.randn(3, 4, device="cuda")
        b = a + 1
        c = a.transpose(0, 1)
        d = a @ a.T
        assert b.shape == (3, 4)
        assert c.shape == (4, 3)
        assert d.shape == (3, 3)


# ---------------------------------------------------------------------------
# jit_cache + COMPILE_ONLY interaction
# ---------------------------------------------------------------------------


def test_jit_cache_compile_only_returns_noop():
    """Under ``COMPILE_ONLY`` ``jit_cache`` must return the no-op kernel so
    downstream call sites can ``compiled_fn(...)`` safely on FakeTensors."""
    import quack.cache
    from quack.cache.jit import _noop_kernel

    calls = []

    @quack.cache.jit_cache
    def _compile_stub(key):
        # Pretend to compile by appending. The real implementation calls
        # cute.compile(); for the test we just track that the body ran.
        calls.append(key)

        # Return a "compiled" function that records its own call.
        def _launched(*a, **kw):
            calls.append(("launched", key))

        return _launched

    quack.cache.COMPILE_ONLY = False
    try:
        # Normal mode: returns the real compiled fn, which records calls.
        real_fn = _compile_stub("k1")
        real_fn("arg")
        assert calls == ["k1", ("launched", "k1")]

        # Compile-only mode: returns _noop_kernel which is a no-op.
        calls.clear()
        quack.cache.COMPILE_ONLY = True
        noop = _compile_stub("k2")
        noop("arg")
        assert noop is _noop_kernel
        assert calls == ["k2"]  # body ran (compile happened) but no launch
    finally:
        quack.cache.COMPILE_ONLY = False


# ---------------------------------------------------------------------------
# CompileOnlyStrictError plumbing
# ---------------------------------------------------------------------------


def test_strict_error_is_a_runtime_error():
    from quack.cache import CompileOnlyStrictError

    assert issubclass(CompileOnlyStrictError, RuntimeError)


def test_strict_mode_wraps_precompile_failures(monkeypatch):
    """``QUACK_COMPILE_ONLY_STRICT=1`` causes ``_precompile_default_config``
    to wrap raised exceptions in ``CompileOnlyStrictError``."""
    monkeypatch.setenv("QUACK_COMPILE_ONLY_STRICT", "1")

    import quack.cache
    from quack.cache import CompileOnlyStrictError
    from quack.gemm_interface import _precompile_default_config

    class _BadFn:
        __name__ = "bad_fn"

        @staticmethod
        def fn(*a, **kw):
            raise TypeError("simulated schema drift")

    quack.cache.COMPILE_ONLY = True
    try:
        with quack.cache.CompileOnlyFakeTensorMode():
            A = torch.randn(8, 8, device="cuda")
            with pytest.raises(CompileOnlyStrictError, match="bad_fn"):
                _precompile_default_config(_BadFn(), A)
    finally:
        quack.cache.COMPILE_ONLY = False


def test_non_strict_mode_swallows_precompile_failures(monkeypatch):
    monkeypatch.delenv("QUACK_COMPILE_ONLY_STRICT", raising=False)

    import quack.cache
    from quack.gemm_interface import _precompile_default_config

    class _BadFn:
        __name__ = "bad_fn"

        @staticmethod
        def fn(*a, **kw):
            raise TypeError("would-be schema drift")

    quack.cache.COMPILE_ONLY = True
    try:
        with quack.cache.CompileOnlyFakeTensorMode():
            A = torch.randn(8, 8, device="cuda")
            # Must not raise; the swallow inside _precompile_default_config
            # is responsible for keeping non-strict runs quiet.
            _precompile_default_config(_BadFn(), A)
    finally:
        quack.cache.COMPILE_ONLY = False


# ---------------------------------------------------------------------------
# Plugin _should_swallow classification
# ---------------------------------------------------------------------------


def test_plugin_should_swallow_classification():
    from quack.cache import CompileOnlyStrictError
    from quack.testing.pytest_plugin import _should_swallow

    # Skip: never swallowed (tests that meant to skip should keep skipping).
    assert not _should_swallow(pytest.skip.Exception)
    # Strict-mode failures: never swallowed (defeating the whole point).
    assert not _should_swallow(CompileOnlyStrictError)
    # Everything else: swallowed under --compile-only.
    assert _should_swallow(RuntimeError)
    assert _should_swallow(TypeError)
    assert _should_swallow(AssertionError)


# ---------------------------------------------------------------------------
# Smoke: module imports / re-imports stay consistent
# ---------------------------------------------------------------------------


def test_repeated_import_is_idempotent():
    """Re-importing ``quack.cache`` must not reset the mutable flags."""
    import quack.cache

    quack.cache.COMPILE_ONLY = True
    try:
        importlib.reload(quack.cache.jit)  # reload a submodule
        # The package-level flag must still be True; reloading a submodule
        # shouldn't reset state owned by __init__.py.
        assert quack.cache.COMPILE_ONLY is True
    finally:
        quack.cache.COMPILE_ONLY = False
