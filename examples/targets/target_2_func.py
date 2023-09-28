from __future__ import annotations

import logging

# ignore DeprecationWarning from `pkg_resources`
logging.captureWarnings(True)


import paddle
import paddle.nn

import paddlefx
import paddlefx.utils

# logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


def func(x, y):
    z = x + y
    return z


@paddlefx.optimize(backend=my_compiler)
def add(a, b):
    d = func(a, b)
    return d


in_a = paddle.rand([1])
in_b = paddle.rand([1])
res = add(in_a, in_b)

print("in_a = ", in_a)
print("in_b = ", in_b)
print("res = ", res)
