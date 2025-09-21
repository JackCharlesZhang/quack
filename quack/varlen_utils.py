# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute

from quack.cute_dsl_utils import ArgumentsBase


# Grouping arguments together that should be passed to __call__
@dataclass
class VarlenArguments(ArgumentsBase):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mTensormaps: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None
