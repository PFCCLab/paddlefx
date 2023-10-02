from __future__ import annotations

import dataclasses

from typing import Callable, Generic, TypeVar

import paddle
import paddle.device

import paddlefx

T = TypeVar("T")


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
class SymbolTable(Generic[T]):
    def __init__(self):
        self._symbol_table: dict[str, T] = {}
        self._inputs: list[T] = []
        self._outputs: tuple[T, ...] = ()

    def __getitem__(self, key: str) -> T:
        return self._symbol_table[key]

    def __setitem__(self, key: str, value: T):
        self._symbol_table[key] = value

    def __iter__(self):
        return iter(self._symbol_table.items())

    @property
    def inputs(self) -> tuple[T, ...]:
        return tuple(self._inputs)

    def add_input(self, key: str, value: T):
        self._inputs.append(value)
        self._symbol_table[key] = value

    @property
    def outputs(self) -> tuple[T, ...]:
        return self._outputs

    @outputs.setter
    def outputs(self, value: tuple[T, ...]):
        self._outputs = value


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
        return self.compile(gl, example_inputs)

    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        symbol_table: SymbolTable = SymbolTable()
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, example_inputs)
            return self.gen_compiled_func(symbol_table)
        except CompilerError as e:
            if self.allow_fallback:
                print(
                    f"CompilerError when compiling graph, useing default forward: {e}"
                )
                self.input_index = 0
                return gl.forward
            raise e
        except AttributeError as e:
            raise AttributeError(
                f"AttributeError when compiling graph, check if you use abstract class"
            ) from e

    def gen_compiled_func(self, symbol_table: SymbolTable):
        raise NotImplementedError("CompilerBase is a abstract class")


class DummyCompiler(CompilerBase):
    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        return gl.forward
