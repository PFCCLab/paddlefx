from __future__ import annotations

import logging

# ignore DeprecationWarning from `pkg_resources`
logging.captureWarnings(True)

import paddle
import paddle._C_ops

import paddlefx

from paddlefx.compiler import TVMCompiler

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
# logging.basicConfig(level=logging.INFO, format="%(message)s")

paddle.seed(0)


def func(x, y):
    z = paddle.add(x, y)
    o = paddle._C_ops.add(z, z)  # type: ignore
    return o


@paddlefx.optimize(backend=TVMCompiler(print_tabular_mode="rich"))
def net(a, b):
    c = func(a, b)
    return c


in_a = paddle.ones([1], dtype='float32')
in_b = paddle.ones([1], dtype='float32')
res = net(in_a, in_b)
print("res = ", res)
