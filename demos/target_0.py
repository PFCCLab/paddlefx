from __future__ import annotations

import logging

# ignore DeprecationWarning from `pkg_resources`
logging.captureWarnings(True)


import paddle
import paddle.nn

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    # gl.graph.print_tabular()
    # return gl.forward

    return print


@paddlefx.optimize(my_compiler)
def add(x, y):
    # print('call add')
    z = x + y
    return


in_a = paddle.rand([1])
in_b = paddle.rand([1])

res = add(in_a, in_b)
print(type(res))
