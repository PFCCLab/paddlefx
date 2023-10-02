from __future__ import annotations

import paddle
import paddle.nn

from utils import check_func


def binary_operator(a, b):
    c = a / b
    d = a * b
    e = c + d
    f = e - a
    g = f > e
    h = g < f
    i = h <= g
    j = i >= i
    k = j == i
    l = j != k
    return l


def binary_inplace_operator(a, b):
    a -= b
    a += b
    a *= b
    a /= b
    a **= b
    a @= b.T
    return a


def inner_func(x, y):
    p = paddle.add(x, y)
    print("call inner")
    q = paddle._C_ops.subtract(x, y)  # type: ignore
    z = p * q
    return z / y


def breakraph_func(a, b):
    print("call func")
    d = inner_func(a, b)
    return d


def test_binary_operator():
    in_a = paddle.rand([1, 24])
    in_b = paddle.rand([8, 24])
    check_func(binary_operator, in_a, in_b)


def test_binary_inplace_operator():
    in_a = paddle.rand([1, 24])
    in_b = paddle.rand([8, 24])
    check_func(binary_inplace_operator, in_a, in_b)


def test_breakraph_func():
    in_a = paddle.rand([8, 8, 1])
    in_b = paddle.rand([8, 8, 16])
    check_func(inner_func, in_a, in_b)
