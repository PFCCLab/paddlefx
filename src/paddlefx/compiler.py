from __future__ import annotations

from typing import Callable

import paddle
import tvm
import tvm.testing

from tvm import te

import paddlefx


class CompilerBase:
    def __init__(self, *, print_tabular: bool = False):
        self.print_tabular = print_tabular

    def __call__(self, gl: paddlefx.GraphLayer, inputs: list):
        if self.print_tabular:
            gl.graph.print_tabular()
        return self.compile(gl, inputs)

    def compile(self, gl: paddlefx.GraphLayer, inputs: list) -> Callable:
        symbol_table = {}
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, inputs)
            return self.gen_compiled_func(symbol_table, inputs)
        except AttributeError as e:
            print(f"AttributeError when compiling graph: {e}")
            return gl.forward

    def gen_compiled_func(self, symbol_table: dict, inputs: list):
        raise NotImplementedError("CompilerBase is a abstract class")


class TVMCompiler(CompilerBase):
    def gen_compiled_func(self, symbol_table: dict, inputs: list):
        tgt = tvm.target.Target(target="llvm", host="llvm")
        s = te.create_schedule(symbol_table["output"].op)
        tvm_func = tvm.build(s, tuple(symbol_table.values())[:3], tgt, name="myadd")

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy()) for arg in args]
            output = tvm.nd.array(args[0].numpy())
            tvm_func(*inputs, output)
            output = paddle.to_tensor(output.asnumpy())
            return output

        return compiled_func

    def compile_placeholder(
        self, node: paddlefx.Node, symbol_table: dict, inputs: list
    ):
        symbol_table[str(node.name)] = te.placeholder(
            inputs.pop(0).shape, name=str(node.name)
        )

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
