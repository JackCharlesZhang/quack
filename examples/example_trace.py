#!/usr/bin/env python3
"""Minimal example: intra-kernel trace profiler in CuTe-DSL.

Run with:    QUACK_TRACE=1 python examples/example_trace.py
Run without: python examples/example_trace.py
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from quack.trace import TraceContext, TraceSession

ITERS = 1000
BLOCK_THREADS = 128
GRID = 4
REGION_NAMES = ("loop_body", "kernel_total")


@cute.kernel
def trace_record_kernel(trace_ptr: Optional[Int64], iters: Int32):
    ctx = TraceContext.create(trace_ptr, region_names=REGION_NAMES)
    ctx.b("kernel_total")
    for i in cutlass.range(iters):
        ctx.b("loop_body")
        ctx.e("loop_body")
    ctx.e("kernel_total")
    ctx.flush()


@cute.jit
def launch(trace_ptr: Optional[Int64], iters: Int32):
    trace_record_kernel(trace_ptr, iters).launch(
        grid=(GRID, 1, 1), block=(BLOCK_THREADS, 1, 1),
    )


def main():
    out_path = "/tmp/cute_dsl_trace.json"

    # sess.ptr is None when QUACK_TRACE != 1 → TraceContext.create becomes a no-op.
    with TraceSession(out_path, grid_size=GRID, block_size=BLOCK_THREADS,
                      region_names=list(REGION_NAMES)) as sess:
        launch(sess.ptr, ITERS)

    if sess.ptr is not None:
        print(f"Open {out_path} in https://ui.perfetto.dev to visualize.")


if __name__ == "__main__":
    main()
