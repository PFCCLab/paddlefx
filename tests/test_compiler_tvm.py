from __future__ import annotations

import paddle
import paddle.nn

from utils import check_func

from paddlefx.compiler.tvm import TVMCompiler

paddle.seed(1234)


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc1 = paddle.nn.Linear(16, 4)
        self.conv1 = paddle.nn.Conv2D(1, 4, 3, 1)

    def forward(self, a, b):
        c = self.fc1(a)
        d = self.conv1(b)
        return c, d


net = SimpleNet()


def test_simple_net():
    in_a = paddle.rand([8, 16])
    in_b = paddle.rand([8, 1, 4, 4])
    check_func(
        net,
        in_a,
        in_b,
        backend=TVMCompiler(full_graph=True, print_tabular_mode="rich"),
    )
