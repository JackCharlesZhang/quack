# Caching and Parallel Compilation

QuACK compiles CUDA kernels at runtime via CuTe DSL. Each compilation generates MLIR IR,
lowers to PTX, and JIT-links — taking ~100ms per kernel. Since a single test run or training
step may trigger dozens of compilations (different dtypes, tile sizes, activations), naive
serial compilation dominates wall-clock time. This doc explains the multi-layer caching and
parallel compilation design that makes it fast.

## Layer 1: In-Memory Cache (`@lru_cache`)

Each `_compile_*` function (e.g. `_compile_gemm` in `gemm.py`, `_compile_gemm_act` in
`gemm_act.py`) is decorated with `@lru_cache(maxsize=None)`. The arguments are all hashable
compile-time parameters: dtypes, tensor majors, tile shape, cluster shape, activation, etc.

Within a single process, calling the same kernel config twice skips compilation entirely.

```
gemm(A, B, D, ...)
  ├─ extract dtypes, majors, tile sizes from real tensors
  └─ _compile_gemm(a_dtype, b_dtype, ..., tile_shape_mn, ...)  # @lru_cache
       └─ ... (compile or cache hit)
```

**Limitation**: `lru_cache` is process-local and lost on restart.

## Layer 2: Filesystem `.so` Cache (`cache_utils.py`)

`compile_and_cache(key, compile_fn)` provides a persistent filesystem cache.

### Cache Key

```python
key = (
    "gemm",           # kernel family
    a_dtype, b_dtype, d_dtype, c_dtype, c_major,  # types
    a_major, b_major, d_major,                     # layout
    tile_shape_mn, cluster_shape_mnk,              # tile config
    pingpong, persistent, ...                       # flags
    device_capacity,                                # SM version
)
```

The key is hashed with SHA-256. The cache directory includes a **source fingerprint** —
a SHA-256 of all `quack/*.py` files plus Python/CUTLASS/TVM-FFI versions. Any source change
invalidates the entire cache, preventing stale `.so` files.

### Cache Miss Flow

```
compile_and_cache(key, compile_fn)
  ├─ hash key → SHA-256 hex
  ├─ Shared lock: check if .so exists → load via dlopen (~1ms) ✓
  │
  ├─ Cache miss → call compile_fn()
  │    └─ cute.compile(...) → MLIR → PTX → binary
  │
  └─ Exclusive lock: export_to_c() → .o → link .so → write to cache
```

### Concurrency Safety

Multiple processes may compile the same key simultaneously (e.g. parallel test workers).
`FileLock` (using `fcntl.flock`) serializes access:

- **Shared lock** for reads — multiple readers can load `.so` concurrently
- **Exclusive lock** for writes — only one writer exports `.so`, others wait
- **Double-check**: after acquiring exclusive lock, re-check if `.so` already exists
  (another process may have written it while we were compiling)

### Environment Controls

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_ENABLED` | `1` | Set to `0` to disable filesystem cache |
| `QUACK_CACHE_DIR` | `/tmp/$USER/quack_cache` | Override cache location |

## Layer 3: Autotuning Result Cache (`autotuner.py`)

The autotuner benchmarks many configs (e.g. ~44 for GEMM) to find the fastest. Results are
cached to disk as JSON via `FileCacheManager` (Triton's cache infrastructure).

```
Autotuner.__call__()
  ├─ Compute tuning key from tensor shapes/strides/dtypes
  ├─ check_disk_cache(key, configs, benchmark_fn)
  │    ├─ Hit: load best config from JSON ✓
  │    └─ Miss: run benchmark_fn(), save JSON
  └─ Call kernel with best config
```

The JSON cache key includes: package version, tuning key (tensor metadata), and all config
strings. Invalidated by changing `__version__` or configs.

| Variable | Default | Description |
|---|---|---|
| `QUACK_CACHE_AUTOTUNING` | unset | Set to `1` to enable disk caching of tuning results |
| `QUACK_FORCE_CACHE_UPDATE` | unset | Set to `1` to ignore cached tuning results |

## Parallel Compilation During Autotuning

When the autotuner encounters a cache miss, it needs to compile all candidate configs before
benchmarking. This is where parallel compilation becomes critical.

### The Constraints

Two facts make naive parallelism impossible:

1. **`cute.compile()` is not thread-safe** — MLIR uses thread-local state, so
   `ThreadPoolExecutor` causes data races.

2. **`fork()` after CUDA init segfaults** — the parent process has already called
   `torch.cuda.init()` (or any CUDA op). `fork()` copies CUDA driver state into the child,
   but driver handles are invalid in the child process, causing segfaults or hangs.
   `multiprocessing.fork` and `os.fork()` both hit this. `mp.set_start_method("spawn")`
   avoids it but `multiprocessing.Pool` with spawn has high per-task overhead.

### Solution: Persistent Subprocess Workers

`Autotuner._precompile()` spawns workers via `subprocess.Popen()` — always a fresh process
(like `spawn`), never `fork`, so no inherited CUDA state.

```
Parent process (CUDA initialized, has real tensors)
  │
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 0
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 1
  └─ ...up to QUACK_COMPILE_WORKERS (default 8)
```

Each worker (`_compile_worker.py`):
1. Sets `COMPILE_ONLY = True` (compilation produces `.so` but never launches kernels)
2. Signals `"READY"` to parent
3. Loops reading tasks from stdin (length-prefixed pickle protocol)
4. Creates `FakeTensor`s matching parent tensor metadata (shape/stride/dtype, no GPU memory)
5. Calls the kernel function under `FakeTensorMode()` → triggers `compile_and_cache()` →
   exports `.so`
6. Stays alive for the next task (amortizes `import quack` overhead, ~2-3s)

The parent does round-robin dispatch, collects acks, then closes stdin to shut workers down.

### Quick-Check Optimization

Before spawning workers, the parent compiles the first config in-process. If this completes
in <0.5s, the `.so` cache is likely warm — skip parallel compilation entirely:

```python
t_check = time.time()
self.fn(*args, **configs[0].all_kwargs())
if time.time() - t_check < 0.5:
    return  # cache is warm, no need for workers
```

### FakeTensors in Workers

Workers don't need real GPU memory. They serialize tensor metadata
(shape, stride, dtype) from the parent, then reconstruct with:

```python
def _make_fake_tensor(meta):
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")
```

Under `FakeTensorMode`, this creates a CPU-side metadata object, no allocation.
Compilation only needs metadata (shapes, strides, dtypes) to generate correct code.

### The CUDA Init Ordering Gotcha

`torch.cuda.init()` must happen *before* entering `FakeTensorMode`:

- `FakeTensorMode` intercepts tensor creation, but CUDA init is a one-time driver call,
  not a tensor op.
- If FakeTensorMode is entered first, CUDA init either doesn't happen or happens in a
  broken state.
- In `_compile_worker.py`, `torch.empty_strided(..., device="cuda")` triggers CUDA init
  on first call — this works because it's a fresh subprocess and FakeTensorMode intercepts
  the allocation after init completes.

## The `COMPILE_ONLY` Flag

`cache_utils.COMPILE_ONLY` is a global boolean (default `False`). When `True`:

1. `compile_and_cache()` still compiles and exports `.so`, but returns `_noop_kernel`
   instead of the real compiled function
2. Early returns in `gemm.py`, `gemm_act.py`, `gemm_dact.py`, `gemm_symmetric.py` after
   compilation prevents reaching runtime code that calls `data_ptr()` (which crashes on
   FakeTensors)

```python
# In gemm.py:
compiled_fn = _compile_gemm(...)

from quack.cache_utils import COMPILE_ONLY
if COMPILE_ONLY:
    return  # skip runtime path (data_ptr, kernel launch, etc.)

# ... runtime code that uses data_ptr(), scalar_arg(), etc.
```

## Two-Pass Test Workflow (`conftest.py`)

The same machinery enables parallel test compilation:

```bash
# Pass 1: compile all kernels in parallel (no GPU memory needed)
pytest tests/test_softmax.py --compile-only -n 64

# Pass 2: run tests (instant .so cache hits)
pytest tests/test_softmax.py
```

### How `--compile-only` Works

`conftest.py:pytest_configure()`:
1. Sets `COMPILE_ONLY = True`
2. Calls `torch.cuda.init()` (before FakeTensorMode — critical ordering)
3. Enters `FakeTensorMode` globally
4. Swallows all test errors via `pytest_runtest_call` hook (we only care about compilation)

### Multi-GPU Round-Robin

When using `pytest-xdist` with multiple GPUs, `conftest.py` assigns workers round-robin:
```python
worker_num = int(worker_id.replace("gw", ""))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_num % len(gpu_ids)]
```

Worker 0 discovers available GPUs and writes to a shared JSON file; other workers read it.

### The GEMM `custom_op` Problem

GEMM functions use `@torch.library.custom_op(device_types="cuda")`, which causes PyTorch
to auto-generate a no-op `register_fake` for `mutates_args` ops. Under `FakeTensorMode`,
this no-op short-circuits the entire function body — the autotuner, compilation, everything.
Result: **0 compilations** for all GEMM variants during `--compile-only`.

Non-GEMM kernels (softmax, rmsnorm, cross_entropy) don't have this problem because their
`register_fake` implementations explicitly call `_compile_*()` when `COMPILE_ONLY` is set.

### The Fix: Explicit `register_fake` for GEMM

Each GEMM `custom_op` gets an explicit `register_fake` that triggers compilation under
`COMPILE_ONLY`. The pattern:

```python
@gemm_out.register_fake
def gemm_out_fake(A, B, out, bias=None, alpha=1.0, ...):
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(A.shape[0], torch.SymInt):
        _precompile_default_config(
            gemm_tuned, A, B, out, C=None, bias=bias, alpha=alpha, ...
        )
```

Key details:

- **`not isinstance(A.shape[0], torch.SymInt)` guard**: Under `torch.compile`, dynamo traces
  with symbolic shapes where `A.shape[0]` is a `SymInt`. We must not compile with SymInts
  (they crash `@lru_cache` and produce invalid code). `COMPILE_ONLY` mode uses concrete shapes.

- **`_precompile_default_config()`**: Calls `autotuned_fn.fn(*args, config=None, **kwargs)`.
  `config=None` selects the default config (128x192 for SM90, 256x256 for SM100). Tests use
  `tuned=False` which bypasses the autotuner and uses the same default, so this is sufficient.

- **`gemm_symmetric_out`** is special: it's not autotuned, so its `register_fake` calls
  `gemm_symmetric_sm90_sm100()` directly with fixed tile parameters.

Ops with explicit `register_fake`:
- `gemm_out`, `gemm_add_out`, `gemm_add_inplace_op` — all route through `gemm_tuned`
- `gemm_act_out` — routes through `gemm_act_tuned`
- `gemm_dact_out` — routes through `gemm_dact_tuned`
- `gemm_gated_out` — routes through `gemm_gated_tuned`
- `gemm_dgated_out` — routes through `gemm_dgated_tuned` (also returns a tensor for
  `colvec_reduce`, so its `register_fake` was already needed for shape inference)
- `gemm_symmetric_out` — calls `gemm_symmetric_sm90_sm100` directly

### Why Early Return After Compilation

The `COMPILE_ONLY` early return in low-level gemm functions (`gemm.py`, `gemm_act.py`, etc.)
prevents reaching runtime code that:
- Calls `data_ptr()` on tensors (crashes on FakeTensors)
- Creates `scalar_arg` from tensor pointers
- Builds runtime `TileSchedulerOptions` with real `data_ptr()` values

```python
# gemm.py — the critical boundary
compiled_fn = _compile_gemm(...)   # ← compilation happens here

from quack.cache_utils import COMPILE_ONLY
if COMPILE_ONLY:
    return                          # ← avoid data_ptr() below

max_active_clusters = get_max_active_clusters(...)
def scalar_arg(scalar, mode):
    if mode == 2:
        return scalar.data_ptr()   # ← would crash on FakeTensor
```

## Complete Data Flow

### Normal Execution (Training/Inference)

```
User calls gemm(A, B, ...)
  └─ gemm_out (custom_op, dispatches to CUDA impl)
       └─ gemm_tuned (autotuner)
            ├─ Cache hit → call fn with best config
            └─ Cache miss → _precompile() + benchmark all configs
                 ├─ Spawn subprocess workers
                 ├─ Workers: FakeTensor + COMPILE_ONLY → .so files
                 ├─ Parent: benchmark each config (instant .so loads)
                 └─ Cache best config
            └─ fn(A, B, out, config=best)
                 └─ gemm_sm90_sm100(...)
                      ├─ _compile_gemm(...) → @lru_cache → compile_and_cache() → .so
                      └─ compiled_fn(real_A, real_B, real_D, ...)
```

### `--compile-only` Test Mode

```
pytest --compile-only
  └─ conftest: COMPILE_ONLY=True, torch.cuda.init(), FakeTensorMode.__enter__()
       └─ Test creates FakeTensors (no GPU memory)
            └─ gemm(A_fake, B_fake, ...)
                 └─ gemm_out (custom_op → dispatches to register_fake)
                      └─ gemm_out_fake(A, B, out, ...)
                           └─ _precompile_default_config(gemm_tuned, ...)
                                └─ gemm_tuned.fn(A, B, out, config=None, ...)
                                     └─ gemm_sm90_sm100(...)
                                          ├─ _compile_gemm(...) → .so exported
                                          └─ COMPILE_ONLY → return (no launch)
```

### `--compile-only` then Normal Run

```
# Pass 1: all .so files written to cache
pytest --compile-only -n 64   # 64 workers, each compiles its test cases

# Pass 2: instant cache hits
pytest
  └─ _compile_gemm(...) → @lru_cache miss → compile_and_cache()
       └─ Shared lock: .so exists → dlopen (~1ms) ✓
```

## Key Files

| File | Role |
|------|------|
| `cache_utils.py` | `.so` cache: `compile_and_cache()`, `FileLock`, `COMPILE_ONLY` flag |
| `compile_utils.py` | `make_fake_tensor()` — symbolic CuTe tensors for compilation |
| `gemm_tvm_ffi_utils.py` | `make_fake_gemm_tensors()`, `cached_compile()`, `compile_gemm_kernel()` |
| `autotuner.py` | `Autotuner._precompile()` — subprocess worker pool, `FileCacheManager` |
| `_compile_worker.py` | Persistent subprocess: `FakeTensorMode` + `COMPILE_ONLY` loop |
| `gemm_interface.py` | `register_fake` for all GEMM `custom_op`s, `_precompile_default_config()` |
| `conftest.py` | `--compile-only` pytest plugin, GPU round-robin, `FakeTensorMode` setup |
| `gemm.py` / `gemm_act.py` / `gemm_dact.py` / `gemm_symmetric.py` | `COMPILE_ONLY` early returns |
