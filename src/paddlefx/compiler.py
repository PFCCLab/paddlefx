from __future__ import annotations

from typing import Any, Callable

import paddle
import tvm
import tvm.testing

from tvm import te

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


class CompilerBase:
    def __init__(self, *, print_tabular: bool = False):
        self.print_tabular = print_tabular
        self.input_index = 0

    def __call__(self, gl: paddlefx.GraphLayer, dummy_inputs: list):
        if self.print_tabular:
            gl.graph.print_tabular()
        return self.compile(gl, dummy_inputs)

    def compile(self, gl: paddlefx.GraphLayer, dummy_inputs: list) -> Callable:
        dummy_outputs = gl.forward(*dummy_inputs)
        symbol_table = {}
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, dummy_inputs)
            self.input_index = 0
            return self.gen_compiled_func(symbol_table, dummy_outputs)
        except AttributeError as e:
            print(f"AttributeError when compiling graph: {e}")
            self.input_index = 0
            return gl.forward

    def gen_compiled_func(self, symbol_table: dict, dummy_outputs: Any):
        raise NotImplementedError("CompilerBase is a abstract class")


class TVMCompiler(CompilerBase):
    def gen_compiled_func(self, symbol_table: dict, dummy_outputs: Any):
        tgt = tvm.target.Target(target="llvm", host="llvm")
        s = te.create_schedule(symbol_table["output"].op)

        tvm_func = tvm.build(
            s,
            [v for k, v in symbol_table.items() if k != "output"],
            tgt,
            name=symbol_table["output"].name,
        )

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy()) for arg in args]
            dummy_output = dummy_outputs[0]
            output = tvm.nd.empty(
                dummy_output.shape, paddle_dtype_to_str(dummy_output.dtype)
            )
            tvm_func(*inputs, output)
            output = paddle.to_tensor(output.asnumpy())
            return (output,)

        return compiled_func

    def compile_placeholder(
        self, node: paddlefx.Node, symbol_table: dict, inputs: list
    ):
        symbol_table[str(node.name)] = te.placeholder(
            inputs[self.input_index].shape,
            paddle_dtype_to_str(inputs[self.input_index].dtype),
            name=str(node.name),
        )
        self.input_index += 1

    def compile_call_function(
        self, node: paddlefx.Node, symbol_table: dict, inputs: list
    ):
        if node.target.__name__ == "add":
            left = symbol_table[str(node.args[0])]
            right = symbol_table[str(node.args[1])]
            symbol_table[str(node.name)] = te.compute(
                left.shape, lambda i, j: left[i, j] + right[i, j], name=str(node.name)
            )
        else:
            raise NotImplementedError

    def compile_output(self, node: paddlefx.Node, symbol_table: dict, inputs: list):
        ret = symbol_table.get(str(node.args[0][0]))
        symbol_table["output"] = ret
