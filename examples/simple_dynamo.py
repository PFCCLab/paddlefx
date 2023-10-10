from __future__ import annotations

import logging

import numpy as np
import paddle
import paddle.nn

import paddlefx

from paddlefx.compiler import DummyCompiler, TVMCompiler

logging.getLogger().setLevel(logging.DEBUG)
static_compier = DummyCompiler(full_graph=True, print_tabular_mode="rich")
compiler = TVMCompiler(full_graph=True, print_tabular_mode="rich")


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
        np.testing.assert_allclose(res, out, rtol=1e-5, atol=1e-6)


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(16, 4)
        self.fc2 = paddle.nn.Linear(16, 1)

    def forward(self, a, b):
        c = self.fc1(a)
        d = self.fc2(b)
        e = paddle.add(c, d)
        return e


net = SimpleNet()


in_a = paddle.rand([8, 16])
in_b = paddle.rand([8, 16])
check_func(net, in_a, in_b, backend=static_compier)
