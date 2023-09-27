from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx


@paddlefx.optimize()
def add(x, y):
    z = x + y
    return z


def func(x, y):
    z = x + y
    return z


@paddlefx.optimize()
def func_add(a, b):
    d = func(a, b)
    return d


def test_add():
    in_a = paddle.rand([1])
    in_b = paddle.rand([1])
    np.testing.assert_allclose(add(in_a, in_b), in_a + in_b)


def test_func_add():
    in_a = paddle.rand([1])
    in_b = paddle.rand([1])
    np.testing.assert_allclose(func_add(in_a, in_b), in_a + in_b)
