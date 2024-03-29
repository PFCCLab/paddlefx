from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

import paddlefx

from paddlefx.compiler import TVMCompiler

paddle.seed(1234)


def inner_func(x, y):
    p = paddle.add(x, y)
    # q = paddle._C_ops.subtract(x, y)  # static unsupported
    z = p * x
    return z / y


def func(a, b):
    d = inner_func(a, b)
    return d


optimized_func = paddlefx.optimize(
    func, backend=TVMCompiler(full_graph=True, print_tabular_mode="rich")
)

x = paddle.rand([4, 6, 1])
y = paddle.rand([4, 6, 224])
for _ in range(10):
    res = optimized_func(x, y)
    res = optimized_func(y, x)
    out = func(y, x)

    np.testing.assert_equal(res.numpy(), out.numpy())
