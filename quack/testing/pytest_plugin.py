# Copyright (c) 2026, Tri Dao.
"""Reusable pytest plugin for QuACK ``--compile-only`` cache warming.

Adds the ``--compile-only`` CLI flag, sets up a session-scoped
:class:`~quack.testing.CompileOnlyFakeTensorMode`, and swallows
test/setup/teardown errors under ``--compile-only`` so the run still finishes
even when individual tests stumble on FakeTensor-incompatible APIs that aren't
on the critical path to a kernel dispatch.

To opt in, add this line to your ``conftest.py``::

    pytest_plugins = ["quack.testing.pytest_plugin"]

After that, ``pytest --compile-only`` populates the persistent ``.o`` cache
without launching kernels (no GPU memory, parallelizable across many CPU
workers). A subsequent normal pytest run hits the disk cache for every kernel
signature warmed by ``--compile-only``.

See :mod:`quack.cache.compile_only` for the underlying mechanism.
"""

from __future__ import annotations

import pytest

from quack.cache import CompileOnlyFakeTensorMode, CompileOnlyStrictError


_fake_mode: CompileOnlyFakeTensorMode | None = None
# Saved value of ``quack.cache.COMPILE_ONLY`` from before the plugin enabled
# compile-only mode, so ``pytest_unconfigure`` restores it instead of
# unconditionally setting False (which would clobber a downstream caller that
# set it True out-of-band before pytest started).
_prev_compile_only: bool | None = None


def _should_swallow(exc_type) -> bool:
    """Should a ``--compile-only`` runtime error be force-passed?

    No for:
    * :class:`pytest.skip.Exception` — explicit skips must keep skipping.
    * :class:`quack.cache.CompileOnlyStrictError` — strict-mode precompile
      failures must surface as test failures, otherwise
      ``QUACK_COMPILE_ONLY_STRICT=1`` would be silently defeated by the
      blanket swallow.
    Yes for everything else: by the time we're under ``--compile-only``,
    the only thing that matters is that the kernel dispatched; downstream
    FakeTensor-incompatible APIs (``.numpy()``, ``assert_close``, etc.)
    are expected to fail and don't represent a regression.
    """
    if issubclass(exc_type, pytest.skip.Exception):
        return False
    if issubclass(exc_type, CompileOnlyStrictError):
        return False
    return True


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help=(
            "Compile all kernels and export the .o cache, skip actual kernel "
            "execution. Uses FakeTensorMode (no GPU memory) so you can run "
            "many xdist workers in parallel. A subsequent normal pytest run "
            "hits the disk cache for every kernel signature warmed here."
        ),
    )


def _is_compile_only(config) -> bool:
    try:
        return bool(config.getoption("--compile-only", default=False))
    except (ValueError, AttributeError):
        return False


def _disable_unused_accelerator_lazy_call() -> None:
    """No-op ``_lazy_call`` for accelerators that aren't available.

    Every ``torch.manual_seed(seed)`` fans out across CUDA, MPS, XPU, MTIA, and
    any custom device. For each *uninitialized* backend, ``_lazy_call`` takes
    a slow path that calls ``traceback.format_stack()`` to record where the
    seed was queued from. Under pytest the call stack is deep, so each
    ``format_stack`` costs ~1 ms; across thousands of tests this is several
    seconds of pure CPU overhead per worker, even though no XPU/MTIA work is
    ever submitted.

    Subtlety: ``torch.xpu/random.py`` and ``torch.cuda/random.py`` do
    ``from . import _lazy_call`` at import time, so we must replace the name
    on the submodule too — patching only the package attribute is not enough.
    """
    import torch

    nop = lambda callable, **kwargs: None  # noqa: E731
    if not torch.xpu.is_available():
        torch.xpu._lazy_call = nop
        torch.xpu.random._lazy_call = nop  # captured via `from . import _lazy_call`
    if not torch.mtia.is_available():
        torch.mtia._lazy_call = nop


def pytest_configure(config):
    """Enter the compile-only context for the duration of the test session."""
    global _fake_mode, _prev_compile_only

    # Speed up manual_seed in real (non-compile-only) runs by short-circuiting
    # the queued-seed path on unavailable accelerators. Harmless under
    # --compile-only too (FakeTensorMode swallows the seed call anyway).
    _disable_unused_accelerator_lazy_call()

    if not _is_compile_only(config):
        return

    import torch

    import quack.cache

    _prev_compile_only = quack.cache.COMPILE_ONLY
    quack.cache.COMPILE_ONLY = True
    if torch.cuda.is_available():
        torch.cuda.init()
    _fake_mode = CompileOnlyFakeTensorMode()
    _fake_mode.__enter__()


def pytest_unconfigure(config):
    """Exit the compile-only context and restore the prior COMPILE_ONLY flag."""
    global _fake_mode, _prev_compile_only
    if _fake_mode is None:
        return
    _fake_mode.__exit__(None, None, None)
    _fake_mode = None
    import quack.cache

    if _prev_compile_only is not None:
        quack.cache.COMPILE_ONLY = _prev_compile_only
        _prev_compile_only = None


# --- Error swallowing under --compile-only --------------------------------
#
# Compile-only runs only care that kernels reach their compile path. Test
# bodies will hit FakeTensor-incompatible APIs further down (numerical
# comparisons via assert_close, .cpu()/.numpy() round-trips, etc.) that we
# don't care about for the cache-warming goal. Swallow those errors so the
# run completes; the real (non-compile-only) pass exercises the assertions.
#
# pytest.skip.Exception is *not* swallowed: tests that genuinely want to skip
# under --compile-only should keep skipping.


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    if not _is_compile_only(item.config):
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and _should_swallow(outcome.excinfo[0]):
        outcome.force_result(None)
