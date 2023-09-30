from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx

from paddlefx.compiler import TVMCompiler

# logging.getLogger().setLevel(logging.DEBUG)


@paddlefx.optimize(backend=TVMCompiler(print_tabular=True))
def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(backend=TVMCompiler(print_tabular=True))
def func(a, b):
    print('\tcall func')
    c = add(a, b)
    d = add(c, c)
    return d


in_a = paddle.rand([3, 4])
in_b = paddle.rand([3, 4])
out = paddle.add(in_a, in_b)

res = add(in_a, in_b)
np.testing.assert_equal(res.numpy(), out.numpy())


def foo(a, b):
    # print('\tcall foo')
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


optimized_foo = paddlefx.optimize(backend=TVMCompiler(print_tabular=True))(foo)

original_res = foo(in_a, in_b)
optimized_res = optimized_foo(in_a, in_b)

np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())

dtype = 'float32'
in_a = paddle.to_tensor([1, 2], dtype=dtype)
in_b = paddle.to_tensor([0, 1], dtype=dtype)


def inplace(a, b):
    # print('\tcall inplace')
    a -= b
    a += b
    a *= b
    a /= b
    a **= b
    a @= b
    return a


optimized_foo = paddlefx.optimize(backend=TVMCompiler(print_tabular=True))(inplace)

original_res = inplace(in_a, in_b)
optimized_res = optimized_foo(in_a, in_b)

np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())


class ExampleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = [paddle.nn.Linear(1, 1), paddle.nn.Linear(1, 1)]

    def forward(self, a, b):
        c = self.fc[0](a[0])
        d = self.fc[1](b[0])
        e = paddle.add(c, d)
        return e


net = ExampleNet()
optimized_func = paddlefx.optimize(backend=TVMCompiler(print_tabular=True))(net)

original_res = net(in_a, in_b)
optimized_res = optimized_func(in_a, in_b)
optimized_res = optimized_func(in_a, in_b)
# TODO(zrr1999): `optimized_res` is the result of running the converted bytecode in the future.
np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())
