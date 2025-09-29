# Copyright (c) 2025, Tri Dao.

import os
import pathlib
from functools import partial, lru_cache
from dataclasses import dataclass, fields
import ctypes

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type
from typing import Union

import importlib.util
import cutlass._mlir.dialects.cute as _cute_ir
from cutlass._mlir import ir


StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                types.extend(obj.__get_mlir_types__())
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        # for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        for name, field in non_constexpr_fields.items():
            # non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            # values = values[n_items:]
            n_items = 1
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    pathlib.Path(filepath).write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    cubin_path = os.getenv("CUTE_CUBIN_PATH", None)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=cubin_path
        )
    output = cute_compile_og(*args, **kwargs)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
        if extract is not None:
            sass = extract(cubin_path, None)
            pathlib.Path(cubin_path).with_suffix(".annotated.sass").write_text(sass)
    return output


class _Pointer(Pointer):
    """Runtime representation of a pointer that can inter-operate with
    various data structures, including numpy arrays and device memory.

    :param pointer: The pointer to the data
    :type pointer: int or pointer-like object
    :param dtype: Data type of the elements pointed to
    :type dtype: Type
    :param mem_space: Memory space where the pointer resides, defaults generic
    :type mem_space: _cute_ir.AddressSpace, optional
    :param assumed_align: Alignment of input pointer in bytes, defaults None
    :type assumed_align: int, optional

    :ivar _pointer: The underlying pointer
    :ivar _dtype: Data type of the elements
    :ivar _addr_space: Memory space of the pointer
    :ivar _assumed_align: Alignment of the pointer in bytes
    :ivar _desc: C-type descriptor for the pointer
    :ivar _c_pointer: C-compatible pointer representation
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        # if expected_py_type is Pointer:
        #     return True
        # elif isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer:
        #     return True
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address

    :param dtype: Data type of the pointer elements
    :type dtype: Type[Numeric]
    :param value: Memory address as integer or ctypes pointer
    :type value: Union[int, ctypes._Pointer]
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :type mem_space: AddressSpace, optional
    :param assumed_align: Alignment in bytes, defaults to None
    :type assumed_align: int, optional
    :return: A pointer object
    :rtype: Pointer

    .. code-block:: python

        import numpy as np
        import ctypes

        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)

        # Get pointer address as integer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)