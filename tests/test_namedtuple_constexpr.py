# Test that NamedTuples with Constexpr fields work with TVM-FFI compilation.

from typing import NamedTuple
from enum import IntEnum

import pytest
import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import mlir_namedtuple

# Ensure the Constexpr converter patch is loaded
import quack.cute_dsl_utils  # noqa: F401


class MyEnum(IntEnum):
    ADD = 0
    MUL = 1


@mlir_namedtuple
class MyArgs(NamedTuple):
    mOut: cute.Tensor
    mA: cute.Tensor
    op: cutlass.Constexpr[int] = 0


@cute.kernel
def _apply_op_add(mOut: cute.Tensor, mA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < mOut.shape[0]:
        mOut[tidx] = mA[tidx] + 1


@cute.kernel
def _apply_op_mul(mOut: cute.Tensor, mA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < mOut.shape[0]:
        mOut[tidx] = mA[tidx] * 2


@cute.jit
def apply_op(args: MyArgs):
    if const_expr(args.op == MyEnum.ADD):
        _apply_op_add(args.mOut, args.mA).launch(grid=(1, 1, 1), block=(128, 1, 1))
    else:
        _apply_op_mul(args.mOut, args.mA).launch(grid=(1, 1, 1), block=(128, 1, 1))


@pytest.mark.parametrize("op", [MyEnum.ADD, MyEnum.MUL])
def test_constexpr_in_namedtuple(op):
    """Constexpr fields are baked in at compile time, passed as None at call time."""
    n = cute.sym_int()
    out_fake = fake_tensor(cute.Float32, (n,), divisibility=1)
    a_fake = fake_tensor(cute.Float32, (n,), divisibility=1)

    compiled = cute.compile(
        apply_op,
        MyArgs(mOut=out_fake, mA=a_fake, op=int(op)),
        options="--enable-tvm-ffi",
    )

    a = torch.ones(32, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    # At call time, constexpr fields must be None (baked into compiled fn)
    compiled(MyArgs(mOut=out, mA=a, op=None))

    expected = (a + 1) if op == MyEnum.ADD else (a * 2)
    torch.testing.assert_close(out, expected)
