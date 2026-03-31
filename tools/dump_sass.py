#!/usr/bin/env python3
"""Dump PTX and SASS of cute-dsl kernels from any script.

Sets CUTE_DSL_KEEP_CUBIN=1 and CUTE_DSL_KEEP_PTX=1, runs the target script,
then disassembles all generated .cubin files with nvdisasm.

Usage::

    python tools/dump_sass.py benchmarks/benchmark_gemm.py -- --mnkl 4096,4096,4096,1
    python tools/dump_sass.py benchmarks/benchmark_gemm.py -o /tmp/sass -- --mnkl 4096,4096,4096,1
    python tools/dump_sass.py benchmarks/benchmark_gemm.py --ptx-only -- --mnkl 4096,4096,4096,1
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_nvdisasm():
    path = shutil.which("nvdisasm")
    if path:
        return path
    for cuda_dir in sorted(Path("/usr/local").glob("cuda*"), reverse=True):
        candidate = cuda_dir / "bin" / "nvdisasm"
        if candidate.is_file():
            return str(candidate)
    return None


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        our_argv, script_args = argv[:idx], argv[idx + 1:]
    else:
        our_argv, script_args = argv, []

    parser = argparse.ArgumentParser(
        description="Dump PTX and SASS of cute-dsl kernels.",
        usage="%(prog)s SCRIPT [-o DIR] [--ptx-only] [-- SCRIPT_ARGS...]",
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("-o", "--output-dir", default="dump_sass_out", help="Output directory")
    parser.add_argument("--ptx-only", action="store_true", help="Skip SASS disassembly")
    args = parser.parse_args(our_argv)

    script = Path(args.script)
    if not script.is_file():
        print(f"Error: {script} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.ptx", "*.cubin", "*.sass"):
        for f in out_dir.glob(ext):
            f.unlink()

    env = os.environ.copy()
    env["CUTE_DSL_KEEP_PTX"] = "1"
    env["CUTE_DSL_KEEP_CUBIN"] = "1"
    env["CUTE_DSL_DUMP_DIR"] = str(out_dir.resolve())

    cmd = [sys.executable, str(script)] + script_args
    print(f"Running: {' '.join(cmd)}")
    print(f"Dump dir: {out_dir.resolve()}\n")
    subprocess.run(cmd, env=env)

    ptx_files = sorted(out_dir.glob("*.ptx"))
    cubin_files = sorted(out_dir.glob("*.cubin"))
    print(f"\nPTX: {len(ptx_files)}, CUBIN: {len(cubin_files)}")
    for f in ptx_files:
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")

    if not args.ptx_only and cubin_files:
        nvdisasm = find_nvdisasm()
        if nvdisasm is None:
            print("nvdisasm not found — skipping SASS disassembly", file=sys.stderr)
        else:
            for cubin in cubin_files:
                sass_path = cubin.with_suffix(".sass")
                result = subprocess.run([nvdisasm, str(cubin)], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  nvdisasm failed: {cubin.name}: {result.stderr.strip()}", file=sys.stderr)
                    continue
                sass_path.write_text(result.stdout)
                print(f"  {sass_path.name}  ({result.stdout.count(chr(10))} lines)")

    print("\nSASS files:")
    for f in sorted(out_dir.glob("*.sass")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
