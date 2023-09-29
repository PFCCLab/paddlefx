from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx

paddle.seed(0)


def add(x, y):
    z = x + y
    return z


def inner_func(x, y):
    p = paddle.add(x, y)
    q = paddle._C_ops.subtract(x, y)
    z = p * q
    return z / y


def func(a, b):
    d = inner_func(a, b)
    return d


def check_func(func, *args):
    comiled_func = paddlefx.optimize(func)
    out = func(*args)
    res = comiled_func(*args)
    np.testing.assert_allclose(res, out)


def test_add():
    in_a = paddle.rand([1, 224])
    in_b = paddle.rand([1, 224])
    check_func(add, in_a, in_b)


def test_func_add():
    in_a = paddle.rand([8, 8, 16])
    in_b = paddle.rand([8, 8, 16])
    check_func(func, in_a, in_b)
