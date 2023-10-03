from __future__ import annotations

import dataclasses

from typing import Callable, Generic, TypeVar

import paddle
import paddle.device

import paddlefx

SymbolT = TypeVar("SymbolT")
ValueT = TypeVar("ValueT")


def paddle_dtype_to_str(dtype: paddle.dtype) -> str:
    if dtype == paddle.float32:
        return "float32"
    elif dtype == paddle.float64:
        return "float64"
    elif dtype == paddle.float16:
        return "float16"
    elif dtype == paddle.int32:
        return "int32"
    elif dtype == paddle.int64:
        return "int64"
    elif dtype == paddle.bool:
        return "bool"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class CompilerError(Exception):
    pass


@dataclasses.dataclass
class SymbolTable(Generic[SymbolT, ValueT]):
    def __init__(self):
        self._symbol_table: dict[str, SymbolT] = {}
        self._inputs: list[SymbolT] = []
        self._params: dict[str, tuple[SymbolT, ValueT]] = {}
        self._outputs: list[SymbolT] = []

    def __getitem__(self, key: str) -> SymbolT:
        return self._symbol_table[key]

    def __setitem__(self, key: str, value: SymbolT):
        self._symbol_table[key] = value

    def __iter__(self):
        return iter(self._symbol_table.items())

    @property
    def inputs(self) -> tuple[SymbolT, ...]:
        return tuple(self._inputs)

    def add_input(self, key: str, value: SymbolT):
        self._inputs.append(value)
        self._symbol_table[key] = value

    @property
    def params(self) -> tuple[tuple[SymbolT, ValueT], ...]:
        return tuple(self._params.values())

    def add_param(self, key: str, value: tuple[SymbolT, ValueT]):
        self._params[key] = value
        self._symbol_table[key] = value[0]

    @property
    def outputs(self) -> tuple[SymbolT, ...]:
        return tuple(self._outputs)

    @outputs.setter
    def outputs(self, value: tuple[SymbolT, ...]):
        self._outputs = list(value)

    @property
    def all_symbols(self) -> list[SymbolT]:
        return self._inputs + [param[0] for param in self.params] + self._outputs


class CompilerBase:
    def __init__(
        self,
        *,
        allow_fallback: bool = False,
        full_graph: bool = False,
        print_tabular_mode: str | None = None,
    ):
        self.allow_fallback = allow_fallback
        self.full_graph = full_graph  # TODO: support full_graph
        self.print_tabular_mode = print_tabular_mode
        self.input_index = 0

    def __call__(self, gl: paddlefx.GraphLayer, example_inputs: list):
        self.input_index = 0
        if self.print_tabular_mode is not None:
            gl.graph.print_tabular(print_mode=self.print_tabular_mode)
        try:
            return self.compile(gl, example_inputs)
        except CompilerError as e:
            if self.allow_fallback:
                print(f"CompilerError when compiling graph, using default forward: {e}")
                self.input_index = 0
                return gl.forward
            raise e

    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        raise NotImplementedError("CompilerBase is a abstract class")


class DummyCompiler(CompilerBase):
    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        return gl.forward
