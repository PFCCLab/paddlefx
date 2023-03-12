from __future__ import annotations

import paddle
import paddle.nn

from paddlefx import symbolic_trace


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc1 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc2 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc3 = paddle.nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        y = paddle.add(x=x, y=x)
        return paddle.nn.functional.relu(x=y)


net = MyNet()
traced_layer = symbolic_trace(net)

example_input = paddle.rand([2, 10])
orig_output = net(example_input)
traced_output = traced_layer(example_input)

assert paddle.allclose(orig_output, traced_output)

print(f"python IR for {type(net).__name__}")
traced_layer.graph.print_tabular()
