from __future__ import annotations

import paddle

from paddle.vision.models import resnet18

from paddlefx import FxGraphViewer, symbolic_trace

net = resnet18()
traced_layer = symbolic_trace(net)

g = FxGraphViewer(traced_layer, "resnet18")
g.get_graph_dot().write_raw("resnet.dot")


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x).clamp(min=0.0, max=1.0)


net = MyNet()
traced_layer = symbolic_trace(net)
g = FxGraphViewer(traced_layer, "MyNet")
g.get_graph_dot().write_raw("mynet.dot")
