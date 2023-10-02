from __future__ import annotations

import dataclasses

from typing import Callable, Generic, TypeVar

import paddle
import paddle.device

import paddlefx

PlaceholderT = TypeVar("PlaceholderT")
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
class SymbolTable(Generic[PlaceholderT, ValueT]):
    def __init__(self):
        self._symbol_table: dict[str, PlaceholderT] = {}
        self._inputs: list[PlaceholderT] = []
        self._weights: dict[str, tuple[PlaceholderT, ValueT]] = {}
        self._outputs: tuple[PlaceholderT, ...] = ()

    def __getitem__(self, key: str) -> PlaceholderT:
        return self._symbol_table[key]

    def __setitem__(self, key: str, value: PlaceholderT):
        self._symbol_table[key] = value

    def __iter__(self):
        return iter(self._symbol_table.items())

    @property
    def inputs(self) -> tuple[PlaceholderT, ...]:
        return tuple(self._inputs + [value[0] for value in self._weights.values()])

    def add_input(self, key: str, value: PlaceholderT):
        self._inputs.append(value)
        self._symbol_table[key] = value

    @property
    def weights(self) -> tuple[ValueT, ...]:
        return tuple(value[1] for value in self._weights.values())

    def add_weight(self, key: str, value: tuple[PlaceholderT, ValueT]):
        self._weights[key] = value
        self._symbol_table[key] = value[0]

    @property
    def outputs(self) -> tuple[PlaceholderT, ...]:
        return self._outputs

    @outputs.setter
    def outputs(self, value: tuple[PlaceholderT, ...]):
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
                getattr(self, f"compile_{node.op}")(
                    gl, node, symbol_table, example_inputs
                )
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
