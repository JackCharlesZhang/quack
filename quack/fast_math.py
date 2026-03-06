# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Integer
from cutlass.cutlass_dsl import dsl_user_op
from cutlass import Int32


class FastDivmod(cute.FastDivmodDivisor):
    """We store the divisor along with the FastDivmodDivisor."""

    @dsl_user_op
    def __init__(
        self,
        divisor: Integer,
        is_power_of_2: bool = None,
        *,
        loc=None,
        ip=None,
    ):
        super().__init__(divisor, is_power_of_2=is_power_of_2, loc=loc, ip=ip)
        self.divisor = divisor

    @dsl_user_op
    def __rdivmod__(self, dividend, *, loc=None, ip=None):
        """Override to return Int32 instead of IntValue for Float, Int arithmetic"""
        quotient, remainder = super().__rdivmod__(dividend, loc=loc, ip=ip)
        return (Int32(quotient), Int32(remainder))

    def __extract_mlir_values__(self):
        """Extract MLIR values for Host->Device transfer."""
        return [self._divisor] + cutlass.extract_mlir_values(self.divisor)

    def __new_from_mlir_values__(self, values):
        """Reconstruct FastDivmodDivisor from MLIR values."""
        new_obj = object.__new__(FastDivmod)
        new_obj._divisor = values[0]
        new_obj.divisor = cutlass.new_from_mlir_values(self.divisor, values[1:])
        return new_obj
