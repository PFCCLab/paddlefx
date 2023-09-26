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
        for node in gl.graph.nodes:
            # TODO(zrr1999): support more ops
            if node.op == "output":
                pass
        return gl.forward
