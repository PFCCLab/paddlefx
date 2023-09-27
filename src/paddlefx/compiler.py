from __future__ import annotations

from typing import Callable

import paddlefx


class CompilerBase:
    def __init__(self, *, print_tabular: bool = False):
        self.print_tabular = print_tabular

    def __call__(self, gl: paddlefx.GraphLayer, inputs: list | tuple):
        if self.print_tabular:
            gl.graph.print_tabular()
        return self.compile(gl, inputs)

    def compile(self, gl: paddlefx.GraphLayer, inputs: list | tuple) -> Callable:
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node)
            return gl.forward  # TODO(zrr1999): return compiled_func
        except NotImplementedError:
            print("AttributeError when compiling graph")
            return gl.forward

    def compile_output(self, node: paddlefx.Node):
        raise NotImplementedError


class NumpyCompiler(CompilerBase):
    pass
