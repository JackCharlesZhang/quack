__version__ = "0.4.1"

import os

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
