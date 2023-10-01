from __future__ import annotations

from typing import Any, Callable

import paddle
import paddle.device

import paddlefx


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


class CompilerBase:
    def __init__(self, *, full_graph=False, print_tabular_mode: str | None = None):
        self.full_graph = full_graph  # TODO: support full_graph
        self.print_tabular_mode = print_tabular_mode
        self.input_index = 0

    def __call__(self, gl: paddlefx.GraphLayer, example_inputs: list):
        self.input_index = 0
        if self.print_tabular_mode is not None:
            gl.graph.print_tabular(print_mode=self.print_tabular_mode)
        return self.compile(gl, example_inputs)

    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        symbol_table: dict[str, Any] = {}
        example_outputs = gl.forward(*example_inputs)
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, example_inputs)
            return self.gen_compiled_func(symbol_table, example_inputs, example_outputs)
        except CompilerError as e:
            print(f"CompilerError when compiling graph, useing default forward: {e}")
            self.input_index = 0
            return gl.forward
        except AttributeError as e:
            raise AttributeError(
                f"AttributeError when compiling graph, check if you use abstract class: {e}"
            )

    def gen_compiled_func(
        self, symbol_table: dict[str, Any], dummy_inputs: list, dummy_outputs: Any
    ):
        raise NotImplementedError("CompilerBase is a abstract class")


class DummyCompiler(CompilerBase):
    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        return gl.forward
