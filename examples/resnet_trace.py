from __future__ import annotations

import paddle
import paddle.nn

from paddle.vision.models import resnet18

from paddlefx import symbolic_trace

net = resnet18()
traced_layer = symbolic_trace(net)

example_input = paddle.rand([2, 3, 224, 224])
orig_output = net(example_input)
traced_output = traced_layer(example_input)

assert paddle.allclose(orig_output, traced_output)

print(f"python IR for {type(net).__name__}")
traced_layer.graph.print_tabular()
