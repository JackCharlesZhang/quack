# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""pytest configuration for quack kernel tests.

Supports:
  --compile-only    Compile all kernels (populating .so cache), skip actual execution.
                    Uses FakeTensorMode (no GPU memory) so you can use many xdist workers.

Two-pass workflow (after changing kernel source):
  pytest tests/test_softmax.py --compile-only -n 64   # parallel compile, no GPU memory
  pytest tests/test_softmax.py                         # instant .so loads

Single-pass workflow (cache already warm):
  pytest tests/test_softmax.py                         # all .so cache hits
"""

import pytest


_compile_only = False
_fake_mode = None


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile all kernels and export .so cache, skip actual kernel execution. "
        "Use with -n N (pytest-xdist) for parallel compilation.",
    )


def pytest_configure(config):
    global _compile_only, _fake_mode
    try:
        _compile_only = config.getoption("--compile-only", default=False)
    except (ValueError, AttributeError):
        _compile_only = False
    if _compile_only:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        import quack.cache_utils

        quack.cache_utils.COMPILE_ONLY = True
        torch.cuda.init()
        _fake_mode = FakeTensorMode()
        _fake_mode.__enter__()


def pytest_unconfigure(config):
    global _fake_mode
    if _fake_mode is not None:
        _fake_mode.__exit__(None, None, None)
        _fake_mode = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """In --compile-only mode, swallow setup errors (e.g. fixtures allocating CUDA tensors)."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """In --compile-only mode, swallow all errors — we only care about compilation."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """In --compile-only mode, swallow teardown errors."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None:
        outcome.force_result(None)
