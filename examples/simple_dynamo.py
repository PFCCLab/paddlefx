from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


@paddlefx.optimize(my_compiler)
def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(my_compiler)
def func(a, b):
    print('\tcall func')
    c = add(a, b)
    d = add(c, c)
    return d


in_a = paddle.rand([3, 4])
in_b = paddle.rand([3, 4])
out = paddle.add(in_a, in_b)
# out = paddle.add(out, out)

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


optimized_foo = paddlefx.optimize(my_compiler)(foo)

original_res = foo(in_a, in_b)
optimized_res = optimized_foo(in_a, in_b)

np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())

dtype = 'float32'
in_a = paddle.to_tensor([1], dtype=dtype)
in_b = paddle.to_tensor([0], dtype=dtype)


def inplace(a, b):
    # print('\tcall inplace')
    a -= b
    a += b
    a *= b
    a /= b
    a **= b
    a @= b
    return a


optimized_foo = paddlefx.optimize(my_compiler)(inplace)

original_res = inplace(in_a, in_b)
optimized_res = optimized_foo(in_a, in_b)

np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())


class ExampleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = [paddle.nn.Linear(1, 1), paddle.nn.Linear(1, 1)]

    def forward(self, a, b):
        c = self.fc[0](a)
        d = self.fc[1](b)
        e = paddle.add(c, d)
        return e


test_obj = ExampleNet()
func = test_obj.forward

optimized_func = paddlefx.optimize(my_compiler)(func)

original_res = func(in_a, in_b)
optimized_res = optimized_func(in_a, in_b)
np.testing.assert_equal(original_res.numpy(), optimized_res.numpy())
