# Parallel Compilation During Autotuning

## The Problem

Autotuning needs to compile many kernel configs (often 50+). Each CuTe DSL compilation
takes ~100ms of IR generation + JIT, so serial compilation dominates tuning time.

Two constraints make naive parallelism impossible:

1. **`cute.compile()` is not thread-safe** — it uses MLIR thread-local state, so
   `ThreadPoolExecutor` is out.
2. **`fork()` after CUDA init segfaults** — the parent process has already called
   `torch.cuda.init()` (or any CUDA op) before autotuning starts. `fork()` copies the
   CUDA driver state into the child, but the driver handles are invalid in the child
   process, causing immediate segfaults or hangs. Python's `multiprocessing.fork` and
   `os.fork()` both hit this. `mp.set_start_method("spawn")` avoids it but
   `multiprocessing.Pool` with spawn has high per-task overhead and doesn't amortize
   import time.

## The Solution: Persistent Subprocess Workers

`autotuner.py:_precompile()` spawns workers via `subprocess.Popen()` — this is always a
fresh process (like `spawn`), never `fork`, so there's no inherited CUDA state.

### Architecture

```
Parent process (CUDA initialized, has real tensors)
  │
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 0
  ├─ subprocess.Popen("python -m quack._compile_worker")  ← worker 1
  └─ ...up to QUACK_COMPILE_WORKERS (default 8)
```

Each worker:
1. Sets `COMPILE_ONLY = True` — compilation produces .so files but never launches kernels
2. Signals `"READY"` back to parent
3. Loops reading tasks from stdin (length-prefixed pickle protocol)
4. Creates `FakeTensor`s matching parent tensor metadata (shape/stride/dtype, no GPU memory)
5. Calls the kernel function under `FakeTensorMode()` → triggers `compile_and_cache()` →
   exports .so
6. Stays alive for the next task (amortizes `import quack` overhead, ~2-3s)

The parent does round-robin dispatch, collects acks, then closes stdin to shut workers down.

### Why FakeTensors

Workers don't need real GPU memory — they only need tensor metadata (shape, stride, dtype)
to drive compilation. `FakeTensorMode` lets us:
- Run many workers without exhausting GPU memory
- Avoid CUDA context creation overhead (FakeTensors are CPU-side metadata)
- Still produce valid .so files since compilation only depends on metadata

### The .so Cache

`cache_utils.py:compile_and_cache()` provides the caching layer:
- Key = hash of (kernel source fingerprint + all compile-time parameters)
- On miss: compile → `export_to_c()` → link .so → store under `QUACK_CACHE_DIR`
- On hit: `dlopen()` the .so (~1ms)
- File locking (`fcntl.flock`) handles concurrent workers writing the same key

After precompilation, the parent benchmarks each config — now every `compile_and_cache()`
call is a cache hit loading from .so, so benchmarking measures only kernel runtime.

### The CUDA Init Gotcha

The critical subtlety: `torch.cuda.init()` must happen *before* `FakeTensorMode` is
entered. FakeTensorMode intercepts tensor creation, but CUDA initialization itself is
not a tensor op — it's a one-time driver call. If you enter FakeTensorMode first, then
try to use CUDA, the init either doesn't happen or happens in a broken state.

This matters in two places:
- `_compile_worker.py:_make_fake_tensor()` calls `torch.empty_strided(..., device="cuda")`
  which triggers CUDA init on first call — this works because it's a fresh subprocess
  and FakeTensorMode intercepts the allocation after CUDA init
- `conftest.py` explicitly calls `torch.cuda.init()` on line 45 *before*
  `FakeTensorMode().__enter__()` on line 47

## Two-Pass Test Workflow

The same machinery supports parallel test compilation:

```bash
# Pass 1: compile all kernels in parallel (no GPU memory needed)
pytest tests/test_softmax.py --compile-only -n 64

# Pass 2: run tests (instant .so cache hits)
pytest tests/test_softmax.py
```

`conftest.py` enables `--compile-only` mode: sets `COMPILE_ONLY=True`, calls
`torch.cuda.init()`, enters `FakeTensorMode`, and swallows all test errors (we only
care that compilation succeeded).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QUACK_COMPILE_WORKERS` | `8` | Max parallel compile workers during autotuning |
| `QUACK_CACHE_ENABLED` | `1` | Enable/disable .so cache |
| `QUACK_CACHE_DIR` | `/tmp/$USER/quack_cache` | Override cache location |
| `QUACK_PRINT_AUTOTUNING` | unset | Set to `1` for verbose autotuning output |
| `QUACK_FORCE_CACHE_UPDATE` | unset | Set to `1` to ignore autotuning result cache |
