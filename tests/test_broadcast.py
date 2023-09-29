from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx

from paddlefx.compiler import TVMCompiler


def add(x, y):
    z = x + y
    return z


def check_func(func, *args):
    comiled_func = paddlefx.optimize(func, backend=TVMCompiler(print_tabular=True))
    out = func(*args)
    res = comiled_func(*args)
    np.testing.assert_allclose(res, out)


def test_broadcast_add():
    in_a = paddle.rand([224, 224])
    in_b = paddle.rand([1, 224])
    check_func(add, in_a, in_b)
