from __future__ import annotations

import logging

import numpy as np
import paddle
import paddle.nn

import paddlefx

from paddlefx.compiler import DummyCompiler, TVMCompiler

logging.getLogger().setLevel(logging.DEBUG)
dummy_compier = DummyCompiler(full_graph=True, print_tabular_mode="rich")
compiler = TVMCompiler(full_graph=True, print_tabular_mode="rich")


def inner_func(x, y):
    p = x + y
    q = paddle._C_ops.subtract(x, y)  # type: ignore
    print(1)
    z = p * q
    return z / y


def breakraph_func(a, b):
    d = inner_func(a, b)
    d = inner_func(a, b)
    # print("call func")
    q = inner_func(a, b)
    return d, q


def check_func(func, *args, backend: None = None):
    if backend is None:
        comiled_func = paddlefx.optimize(func)
    else:
        comiled_func = paddlefx.optimize(func, backend=backend)
    out = func(*args)
    res = comiled_func(*args)
    if isinstance(out, tuple):
        for i in range(len(res)):
            np.testing.assert_allclose(res[i], out[i])
    else:
        np.testing.assert_allclose(res, out)


in_a = paddle.rand([8, 8, 16])
in_b = paddle.rand([8, 1, 16])
check_func(inner_func, in_a, in_b, backend=compiler)


# dtype = 'float32'
# in_a = paddle.to_tensor([1, 2], dtype=dtype)
# in_b = paddle.to_tensor([0, 1], dtype=dtype)


# def inplace(a, b):
#     # print('\tcall inplace')
#     a -= b
#     a += b
#     a *= b
#     a /= b
#     a **= b
#     a @= b
#     return a


# optimized_foo = paddlefx.optimize(backend=compiler)(
#     inplace
# )

# original_res = inplace(in_a, in_b)
# optimized_res = optimized_foo(in_a, in_b)

# np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())
