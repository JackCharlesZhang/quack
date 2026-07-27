# Copyright (c) 2026, Tri Dao.
"""Per-operand transforms for GEMM mainloops (RS A-operand produces today;
TransformB smem rewrites later). Ported from the transformA branch onto the
interleaved RS mainloop's copy_block seam."""

from quack.operand_transform.frontend import (  # noqa: F401
    ATransformMod,
    PackedInput,
    a_transform,
    w4_transform,
)
from quack.operand_transform.transform_a import (  # noqa: F401
    TransformA,
    TransformAValue,
    TransformAW4,
)
