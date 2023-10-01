from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

import paddlefx

from paddlefx.compiler.tvm import TVMCompiler

paddle.seed(0)


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = [paddle.nn.Linear(16, 1), paddle.nn.Linear(16, 4)]

    def forward(self, a, b):
        c = self.fc[0](a)
        d = self.fc[1](b)
        e = paddle.add(c, d)
        return e


net = SimpleNet()


def check_func(func, *args):
    comiled_func = paddlefx.optimize(
        func, backend=TVMCompiler(print_tabular_mode="rich")
    )
    out = func(*args)
    res = comiled_func(*args)
    np.testing.assert_allclose(res, out)


def test_simple_net():
    in_a = paddle.rand([8, 16])
    in_b = paddle.rand([8, 16])
    check_func(net, in_a, in_b)
