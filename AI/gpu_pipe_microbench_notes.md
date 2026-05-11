# Notes: SM-local Pipe Throughput — SMEM, SHFL, REDUX

These measurements use two matching microbenchmarks:

- CUDA: `microbenchmarks/gpu_pipe_microbench.cu`
- CuTe-DSL: `microbenchmarks/gpu_pipe_microbench.py`

Both run one CTA per SM, time the hot loop with `clock64()`, and use the same interleaved kernel shape so we can compare standalone and mixed instruction streams.  The SMEM access pattern is a conflict-free `uint4` per lane: each warp touches a contiguous 512 B region, split into four aligned 128 B shared-memory transactions.

## Counting convention

For shared memory, the byte count is literal: `ld/st.shared.v4.u32` moves 16 B per lane.

For shuffle, I count useful data delivered:

```text
1 full-warp 32-bit SHFL = 32 lanes * 4 B = 128 useful B
```

This is the right normalization for modeling contention with SMEM throughput.  Counting it as read+write, i.e. 8 B/lane, is a conceptual register-file accounting, but it does not match the observed SMEM-competition model as well.

For `redux.sync.add.s32`, the primary unit is warp-instructions per clock per SM.  The benchmark also prints an input-byte rate, `32 lanes * 4 B`, but REDUX does not behave like a simple SMEM-bandwidth consumer.

## H100 / SM90 results

Representative results with `--iters 50000 --threads 256`:

```text
SMEM read  peak: ~128 B/clk/SM
SMEM write peak: ~128 B/clk/SM
SHFL peak      : ~1.00 warp-inst/clk/SM = ~128 useful B/clk/SM
REDUX peak     : ~0.50 warp-inst/clk/SM
```

The CuTe-DSL version reproduces the CUDA results nearly exactly: SMEM and SHFL match at ~128 B/clk/SM, REDUX matches at ~0.5 warp-inst/clk/SM, and mixed kernels show the same overlap/competition behavior.

## Does SHFL compete with SMEM?

Yes.  A balanced mixed kernel with one SMEM `uint4` op per lane and four 32-bit SHFLs per interleave step gives roughly:

```text
read  + shfl: total ~120 B/clk/SM, obs/sum ~= 1.06
write + shfl: total ~122 B/clk/SM, obs/sum ~= 1.05
```

Here `obs/sum ~= 1` means the mixed runtime is close to adding standalone SMEM time and standalone SHFL time.  In other words, SHFL and SMEM do not overlap like independent resources; they contend for a shared/nearby SM-local data-movement pipe.  The likely common bottleneck is the MIO pipe / MIO issue path: SMEM `LDS`/`STS` are LSU/L1TEX operations, while SHFL does not touch SMEM banks, but both appear to consume the same SM-local MIO-style movement/issue/writeback resource.

Rule of thumb:

```text
1 full-warp 32-bit SHFL ~= 128 B of SMEM-pipe pressure
                         ~= 4 B/lane
```

So a single 32-bit value exchanged through SMEM costs a store plus a load:

```text
SMEM round trip = 4 B/lane store + 4 B/lane load = 8 B/lane
```

That is roughly the same pipe pressure as two 32-bit SHFL instructions.  If an exchange takes three or more SHFLs but could be implemented as one conflict-free SMEM store/load round trip, SMEM may be better in throughput pressure, though SHFL can still win on latency and simplicity.

## Does REDUX compete with SMEM?

REDUX is different.  Standalone throughput is about half a warp instruction per clock per SM:

```text
1 REDUX.SUM.S32 warp instruction ~= 2 clocks
```

Compared to a 128 B/clk/SM SMEM baseline, that standalone time corresponds to about:

```text
~256 B / warp REDUX = ~8 B/lane
```

But mixed kernels show REDUX does not simply consume SMEM bandwidth.  It overlaps well with SMEM reads:

```text
read + redux, ratio 4: obs/ind ~= 1.0
```

and only partially interferes with SMEM writes:

```text
write + redux, ratio 4: obs/ind ~= 1.2-1.25
```

So the useful model is:

- SHFL: model as SMEM-pipe pressure, about `4 B/lane` per 32-bit SHFL.
- REDUX: model as its own slower collective pipe, about `0.5 warp-inst/clk/SM`, mostly overlapping with SMEM reads and partially interfering with writes.

## Reproducing

CUDA:

```bash
nvcc -O3 -std=c++17 -arch=native microbenchmarks/gpu_pipe_microbench.cu -o gpu_pipe_microbench
./gpu_pipe_microbench --iters 50000 --threads 256
```

CuTe-DSL:

```bash
python microbenchmarks/gpu_pipe_microbench.py \
  --iterations 50000 --threads 256 --rep 1 --warmup 0
```

The CuTe-DSL benchmark uses the same flags conceptually (`smem_read`, `smem_write`, `shuffle`, `redux`, and per-op counts) so it is a convenient template for adding more SM-local instructions later.
