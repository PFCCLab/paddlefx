from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

from paddle.vision.models import resnet18

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    print(gl.get_source())
    gl.graph.print_tabular(print_mode="rich")
    return gl.forward


net = resnet18()
optimized_net = paddlefx.optimize(backend=my_compiler)(net)

x = paddle.rand([1, 3, 224, 224])
out = net(x)
res = optimized_net(x)

np.testing.assert_equal(res.numpy(), out.numpy())
