from __future__ import annotations

import logging

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

import paddlefx

from paddlefx.compiler import TVMCompiler

paddle.seed(0)

logging.getLogger().setLevel(logging.DEBUG)


def inner_func(x, y):
    p = paddle.add(x, y)
    q = paddle._C_ops.subtract(x, y)
    z = p * q
    return z / y


def func(a, b):
    d = inner_func(a, b)
    return d


optimized_net = paddlefx.optimize(func, backend=TVMCompiler(print_tabular=True))

x = paddle.rand([1, 224])
y = paddle.rand([1, 224])
out = func(x, y)
res = optimized_net(x, y)

np.testing.assert_equal(res.numpy(), out.numpy())
