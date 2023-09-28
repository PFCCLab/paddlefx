from __future__ import annotations

import logging

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

import paddlefx

from paddlefx.compiler import TVMCompiler

logging.getLogger().setLevel(logging.DEBUG)


def inner_func(x, y):
    p = x + y
    q = x / y
    return p - q


def func(x, y):
    res = inner_func(x, y)
    return res


optimized_net = paddlefx.optimize(func, backend=TVMCompiler(print_tabular=True))

x = paddle.rand([1, 224])
y = paddle.rand([1, 224])
out = func(x, y)
res = optimized_net(x, y)

np.testing.assert_equal(res.numpy(), out.numpy())
