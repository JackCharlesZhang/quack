# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent .so cache for CuTe DSL compiled kernels.

Compiled kernels are exported as shared libraries (.so) via export_to_c.
On subsequent runs the .so is loaded via dlopen (~1ms) instead of
re-generating IR + re-JIT'ing (~100ms per kernel).

Controls:
  QUACK_CACHE_ENABLED=0       — disable persistent .so cache (default: enabled)
  QUACK_CACHE_DIR=path        — override default cache directory
"""

import ctypes
import fcntl
import hashlib
import os
import pickle
import sys
import tempfile
import time
from distutils.ccompiler import CCompiler, new_compiler
from functools import lru_cache
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: str | None = os.getenv("QUACK_CACHE_DIR", None)
COMPILE_ONLY: bool = False


def _noop_kernel(*args, **kwargs):
    pass


def get_cache_path() -> Path:
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "quack_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash all quack Python sources plus runtime ABI stamps into a short fingerprint."""
    quack_root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    for src in sorted(quack_root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(quack_root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pre-load cute DSL runtime libraries with RTLD_GLOBAL so that .so files
# can resolve symbols like _cudaLibraryLoadData.
# ---------------------------------------------------------------------------

_runtime_libs_loaded = False


def _ensure_runtime_libs():
    global _runtime_libs_loaded
    if _runtime_libs_loaded:
        return
    for path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        if Path(path).exists():
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    _runtime_libs_loaded = True


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# Persistent cache
# ---------------------------------------------------------------------------

EXPORT_FUNC_NAME = "func"
LOCK_TIMEOUT = 60

_compiler: CCompiler | None = None


def _get_compiler() -> CCompiler:
    global _compiler
    if _compiler is None:
        _compiler = new_compiler()
    return _compiler


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _load_from_so(so_path: Path) -> object:
    _ensure_runtime_libs()
    m = cute.runtime.load_module(str(so_path), enable_tvm_ffi=True)
    return m[EXPORT_FUNC_NAME]


def _export_to_so(compiled_fn, so_path: Path) -> None:
    so_path.parent.mkdir(parents=True, exist_ok=True)
    obj_path = so_path.with_suffix(".o")
    compiled_fn.export_to_c(
        object_file_path=str(obj_path),
        function_name=EXPORT_FUNC_NAME,
    )
    _get_compiler().link_shared_object([str(obj_path)], str(so_path))
    obj_path.unlink()


def compile_and_cache(key: tuple, compile_fn):
    """Check persistent .so cache; on miss, call compile_fn() and export.

    Args:
        key: Hashable tuple identifying this compilation (include source fingerprint).
        compile_fn: Zero-arg callable that returns a compiled CuTe DSL function.

    Returns:
        The compiled function (either loaded from .so or freshly compiled).
    """
    if not CACHE_ENABLED:
        compiled_fn = compile_fn()
        return _noop_kernel if COMPILE_ONLY else compiled_fn

    cache_path = get_cache_path() / _compute_source_fingerprint()
    cache_path.mkdir(parents=True, exist_ok=True)

    sha = _key_to_hash(key)
    so_path = cache_path / f"{sha}.so"
    lock_path = cache_path / f"{sha}.lock"

    # Try loading under shared lock
    try:
        with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
            if so_path.exists():
                if COMPILE_ONLY:
                    return _noop_kernel
                return _load_from_so(so_path)
    except RuntimeError:
        pass

    # Cache miss — compile
    compiled_fn = compile_fn()

    # Export under exclusive lock
    try:
        with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
            if not so_path.exists():
                _export_to_so(compiled_fn, so_path)
    except Exception as e:
        print(f"quack cache: export failed for key {sha}: {e}")

    return _noop_kernel if COMPILE_ONLY else compiled_fn
