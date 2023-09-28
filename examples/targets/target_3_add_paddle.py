from __future__ import annotations

import logging

# ignore DeprecationWarning from `pkg_resources`
logging.captureWarnings(True)

import paddle
import paddle._C_ops

import paddlefx

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
# logging.basicConfig(level=logging.INFO, format="%(message)s")

paddle.seed(0)


def my_compiler(gl: paddlefx.GraphLayer, example_inputs=None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


def func(x, y):
    z = paddle.add(x, y)
    o = paddle._C_ops.add(z, z)
    return o


@paddlefx.optimize(backend=my_compiler)
def net(a, b):
    c = func(a, b)
    return c


in_a = paddle.ones([1], dtype='float32')
in_b = paddle.ones([1], dtype='float32')
res = net(in_a, in_b)
print("res = ", res)
