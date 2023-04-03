from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


@paddlefx.optimize(my_compiler)
def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(my_compiler)
def func(a, b):
    print('\tcall func')
    c = add(a, b)
    d = add(c, c)
    return d


in_a = paddle.rand([3, 4])
in_b = paddle.rand([3, 4])
out = paddle.add(in_a, in_b)

res = add(in_a, in_b)
np.testing.assert_equal(res.numpy(), out.numpy())
