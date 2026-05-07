"""Shared helpers for triton perf_report-based benchmarks."""

import os

import pandas as pd
from triton.testing import Benchmark


def run_and_print(mark, save_path=None):
    """Run a triton ``Mark`` (from ``perf_report``) and print/save results.

    Each runner is expected to return a ``dict[str, Any]`` mapping stat name to
    value, e.g. ``{"ms": 0.123, "GB/s": 1234}``. All providers in a benchmark
    must return the same set of keys. Values are written through unchanged --
    rounding/formatting is the caller's responsibility.

    Output columns are ``x_names + [f"{line_name} ({stat})" for ...]``.
    """
    benchmarks = mark.benchmarks if isinstance(mark.benchmarks, list) else [mark.benchmarks]
    for bench in benchmarks:
        df = _run_one(mark.fn, bench)
        print(bench.plot_name + ":")
        print(df.to_string())
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), index=False)


def _run_one(fn, bench: Benchmark) -> pd.DataFrame:
    x_names = list(bench.x_names)
    rows = []
    stat_keys = None  # locked in from the first runner result
    for x in bench.x_vals:
        if not isinstance(x, (list, tuple)):
            x = [x] * len(x_names)
        x_args = dict(zip(x_names, x))
        row = list(x)
        for line_val in bench.line_vals:
            stats = fn(**x_args, **{bench.line_arg: line_val}, **bench.args)
            if not isinstance(stats, dict):
                raise TypeError(f"runner must return dict[str, Any], got {type(stats).__name__}")
            if stat_keys is None:
                stat_keys = list(stats.keys())
            elif list(stats.keys()) != stat_keys:
                raise ValueError(f"runner returned keys {list(stats.keys())}, expected {stat_keys}")
            row.extend(stats[k] for k in stat_keys)
        rows.append(row)
    cols = list(x_names) + [
        f"{name} ({stat})" for name in bench.line_names for stat in (stat_keys or [])
    ]
    return pd.DataFrame(rows, columns=cols)
