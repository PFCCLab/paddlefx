from __future__ import annotations

import functools

import paddle
import paddle.nn

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    # gl.graph.print_tabular()
    # return gl.forward

    xx = functools.partial(print, "aaaaa")
    return xx


@paddlefx.optimize(my_compiler)
def add(x, y):
    # print('call add')
    z = x + y
    return z


in_a = paddle.rand([3, 4])
in_b = paddle.rand([3, 4])

res = add(in_a, in_b)
print(type(res))
