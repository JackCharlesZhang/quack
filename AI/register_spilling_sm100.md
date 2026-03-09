# Fixing Register Spilling on SM100 (setmaxregister)

## Problem

The `gemm_dgated` kernel with `colvec_reduce=True` and `gather_A=True` had 90 STL/LDL (local memory) instructions — register spilling. Without `colvec_reduce`, there were 0 spills.

## Root Cause

With `gather_A=True`, the SM100 GEMM kernel uses **3 warp groups** (12 warps, 384 threads). The register budget is `504 / 3 = 168` registers per thread by default. The `colvec_reduce` epilogue exceeds this: it keeps a persistent `tDrColVecReduce` accumulator plus many live register tensors in `epi_visit_subtile` (`tRS_rD`, `tRS_rdXY_f32x2`, `tRS_rOut`, `tRS_rD_scaled`, `tDrColVec`).

Without `gather_A`, there are only 2 warp groups → `512 / 2 = 256` registers per thread (the max), so no spilling.

## Fix

Use `cute.arch.setmaxregister_increase/decrease` in `gemm_sm100.py` to redistribute registers across warp groups, **only when `gather_A=True`** (3 WGs):

- **Epilogue warps** (0–3, 1 WG): `setmaxregister_increase(256)` — full budget for compute-heavy epilogue
- **All other warps** (MMA, load, scheduler, 2 WGs): `setmaxregister_decrease(120)` — they use tmem/TMA and need few registers

Budget check: `256 + 120 + 120 = 496 ≤ 504`. Values must be divisible by 8.

## How to Verify

Get SASS from the compiled kernel and count local memory instructions:

```python
from quack.cache_utils import get_cache_path, _compute_source_fingerprint
cache_dir = get_cache_path() / _compute_source_fingerprint()
```

```python
# Extract cubin from .so
data = open("kernel.so", "rb").read()
import re
positions = [m.start() for m in re.finditer(b'\x7fELF', data)]
open("kernel.cubin", "wb").write(data[positions[1]:])
```

```bash
nvdisasm -gi kernel.cubin | grep -cE '\bSTL\b|\bLDL\b'
```

## Constraints for setmaxregister on SM100

- Total register budget = `(max_regs_per_thread) × num_warp_groups`, max is 512
- Must be divisible by `num_warp_groups` (e.g. 504 for 3 WGs, 512 for 2 WGs)
- Each WG's allocation must be divisible by 8
- Too many regs can cause hangs — stay within the total budget
