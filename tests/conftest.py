# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""pytest configuration for quack kernel tests.

Supports:
  --compile-only    Compile all kernels (populating .o cache), skip actual execution.
                    Uses FakeTensorMode (no GPU memory) so you can use many xdist workers.
                    Works without a GPU if QUACK_ARCH and CUTE_DSL_ARCH are set.

Two-pass workflow (after changing kernel source):
  pytest tests/test_softmax.py --compile-only -n 64   # parallel compile, no GPU memory
  pytest tests/test_softmax.py                         # instant .o loads

CPU-only compilation (no GPU needed):
  QUACK_ARCH=90 CUTE_DSL_ARCH=sm_90a pytest tests/ --compile-only -n 64

Single-pass workflow (cache already warm):
  pytest tests/test_softmax.py                         # all .o cache hits

Multi-GPU with xdist:
  pytest tests/ -n 4                                   # workers round-robin across GPUs
"""

import os
import subprocess
import json
import time
import logging
import tempfile
from pathlib import Path
from getpass import getuser

import pytest


_compile_only = False
_fake_mode = None


def pytest_addoption(parser):
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile all kernels and export .o cache, skip actual kernel execution. "
        "Use with -n N (pytest-xdist) for parallel compilation.",
    )


def _get_gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",")]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()
    except (FileNotFoundError,):
        pass
    logging.warning("Failed to get gpu ids, use default '0'")
    return ["0"]


def _setup_worker_logging(worker_id, tmp):
    """Configure per-worker file logging for easier debugging of parallel runs."""
    log_file = tmp / f"tests_{worker_id}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.info("Worker %s logging to %s", worker_id, log_file)


# Per-session bookkeeping for the xdist worker-crash retry hook below.
_crash_retried: set[str] = set()


def pytest_handlecrashitem(crashitem, report, sched):
    """Re-queue a worker-crashed test on a fresh worker.

    Background. cute.compile() retains one ir.Context (~1.9 MB of MLIR IR)
    per call in C++ state we can't reach from Python (cutlass#3062 fixed the
    related thread-pool leak; the Context-object leak is unfixed upstream).
    After several minutes a worker's RSS exceeds the apptainer cgroup limit
    and the OOM-killer takes it. --max-worker-restart spawns a fresh
    replacement worker, but xdist still marks the test that was in flight
    when the death happened as 'failed' (see xdist.dsession.handle_crashitem,
    which synthesises a TestReport with longrepr only and no excinfo).

    pytest-rerunfailures' --only-rerun matches against excinfo and so cannot
    see worker-crash failures. This hook is the xdist-specific entry point:
    it re-queues the crashed test via the scheduler's mark_test_pending API
    and downgrades the synthetic crash report from 'failed' to 'skipped' so
    the run isn't marked red on the leak alone. The retry produces its own
    pass/fail report; if a test deterministically crashes any worker, the
    second crash isn't retried and the failure stands.
    """
    if crashitem in _crash_retried:
        # Already gave this nodeid one retry. If we're here again the bug is
        # in the test (or in code the test exercises), not in cumulative RSS.
        # Let xdist log the failure normally.
        return
    _crash_retried.add(crashitem)

    try:
        sched.mark_test_pending(crashitem)
    except (NotImplementedError, AttributeError, ValueError):
        # loadscope / each / unexpected scheduler shape: don't try to be
        # clever, just let the failure go through.
        return

    # Mutate the synthetic crash report so it isn't counted as a failure.
    # outcome='skipped' is the standard pytest signal; longrepr for skipped
    # tests is conventionally a (path, lineno, reason) tuple.
    fspath = report.location[0] if getattr(report, "location", None) else ""
    report.outcome = "skipped"
    report.longrepr = (
        fspath,
        0,
        f"worker crashed; re-queued for one retry on a fresh worker "
        f"({crashitem}). See cutlass#3062 background in tests/conftest.py.",
    )


def pytest_configure(config):
    global _compile_only, _fake_mode

    try:
        _compile_only = config.getoption("--compile-only", default=False)
    except (ValueError, AttributeError):
        _compile_only = False

    # Assign GPUs to xdist workers round-robin (skip for CPU-only compile)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id and not (_compile_only and not _has_gpu()):
        tmp = Path(tempfile.gettempdir()) / getuser() / "quack_tests"
        tmp.mkdir(parents=True, exist_ok=True)
        worker_num = int(worker_id.replace("gw", ""))
        cached_gpu_ids = tmp / "gpu_ids.json"
        if worker_num == 0:
            gpu_ids = _get_gpu_ids()
            with cached_gpu_ids.open(mode="w") as f:
                json.dump(gpu_ids, f)
        else:
            while not cached_gpu_ids.exists():
                time.sleep(0.1)
            with cached_gpu_ids.open() as f:
                gpu_ids = json.load(f)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
        _setup_worker_logging(worker_id, tmp)

    if _compile_only:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        import quack.cache_utils

        quack.cache_utils.COMPILE_ONLY = True
        if torch.cuda.is_available():
            torch.cuda.init()
        _fake_mode = FakeTensorMode()
        _fake_mode.__enter__()


def _has_gpu():
    """Check for GPU without initializing CUDA."""
    import torch

    return torch.cuda.is_available()


def pytest_unconfigure(config):
    global _fake_mode
    if _fake_mode is not None:
        _fake_mode.__exit__(None, None, None)
        _fake_mode = None


def pytest_collection_modifyitems(config, items):
    """Deselect ``use_compile=True`` parametrizations under ``--compile-only``.

    Compile-only runs exist to populate the cute kernel .o cache via FakeTensorMode.
    The ``use_compile=True`` parametrizations wrap the kernel entry point in
    ``torch.compile(...)``, but the cute kernel is already an opaque ``cute_op``
    (see ``quack/dsl/torch_library_op.py``), so Dynamo+Inductor add no cache
    coverage — both branches hit the same ``cute.compile`` path through the fake
    impl. Inductor's pre-codegen alignment check (``_inductor/utils.py:tensor_is_aligned``)
    also calls ``data_ptr()`` on the FakeTensor input, which emits a noisy
    deprecation warning. Deselecting these parametrizations removes the warning and
    saves the redundant Inductor compile work. The real GPU pass (no
    ``--compile-only``) still exercises ``torch.compile`` for coverage.
    """
    if not _compile_only:
        return

    deselected = [
        item
        for item in items
        if getattr(item, "callspec", None) is not None
        and item.callspec.params.get("use_compile") is True
    ]
    if not deselected:
        return
    config.hook.pytest_deselected(items=deselected)
    deselected_ids = {id(item) for item in deselected}
    items[:] = [item for item in items if id(item) not in deselected_ids]


def pytest_collection_finish(session):
    """Print a summary of collected tests grouped by file and function."""
    if not session.items:
        return
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for item in session.items:
        file_name = item.location[0]
        func_name = item.originalname if hasattr(item, "originalname") else item.name
        counts[file_name][func_name] += 1
    summary = {f: dict(funcs) for f, funcs in sorted(counts.items())}
    total = len(session.items)
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"Collected {total} tests: {json.dumps(summary, indent=2)}"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """In --compile-only mode, swallow setup errors (e.g. fixtures allocating CUDA tensors)."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and not issubclass(outcome.excinfo[0], pytest.skip.Exception):
        outcome.force_result(None)


def _is_oom(exc_type, exc_val):
    """Check if an exception is a CUDA out-of-memory error."""
    import torch

    if issubclass(exc_type, torch.OutOfMemoryError):
        return True
    if issubclass(exc_type, RuntimeError) and "out of memory" in str(exc_val).lower():
        return True
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """In --compile-only mode, swallow all errors — we only care about compilation.
    In normal mode, retry once on CUDA OOM after freeing GPU memory.
    """
    if not _compile_only:
        outcome = yield
        if outcome.excinfo is not None and _is_oom(*outcome.excinfo[:2]):
            import gc
            import torch

            logging.warning("OOM in %s, freeing GPU memory and retrying once", item.nodeid)
            gc.collect()
            torch.cuda.empty_cache()
            outcome.force_result(None)
            item.runtest()
        return
    outcome = yield
    if outcome.excinfo is not None and not issubclass(outcome.excinfo[0], pytest.skip.Exception):
        outcome.force_result(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """In --compile-only mode, swallow teardown errors."""
    if not _compile_only:
        yield
        return
    outcome = yield
    if outcome.excinfo is not None and not issubclass(outcome.excinfo[0], pytest.skip.Exception):
        outcome.force_result(None)
