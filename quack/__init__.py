__version__ = "0.3.11"

import os

# NVIDIA/cutlass#3161: fix duplicate .text section flags in CuTeDSL-emitted .o
# files before anything else imports / compiles. Must run before the first
# cute.compile call; see quack.cute_dsl_elf_fix for details.
import quack.cute_dsl_elf_fix

quack.cute_dsl_elf_fix.patch()

from quack.rmsnorm import rmsnorm  # noqa: E402
from quack.softmax import softmax  # noqa: E402
from quack.cross_entropy import cross_entropy  # noqa: E402
from quack.rounding import RoundingMode  # noqa: E402


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    quack.cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
]
