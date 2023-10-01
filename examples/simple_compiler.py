from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

import paddlefx

from paddlefx.compiler import TVMCompiler

# logging.getLogger().setLevel(logging.DEBUG)


def inner_func(x, y):
    p = paddle.add(x, y)
    q = paddle._C_ops.subtract(x, y)
    z = p * q
    return z / y


def func(a, b):
    d = inner_func(a, b)
    d = inner_func(a, d)
    return d


optimized_func = paddlefx.optimize(func, backend=TVMCompiler(print_tabular=True))

x = paddle.rand([4, 6, 1])
y = paddle.rand([4, 6, 24])
out = func(y, x)
res = optimized_func(x, y)
res = optimized_func(y, x)

np.testing.assert_equal(res.numpy(), out.numpy())
