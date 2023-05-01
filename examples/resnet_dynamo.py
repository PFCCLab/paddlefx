from __future__ import annotations

import numpy as np
import paddle
import paddle.nn

from paddle.vision.models import resnet18

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: list[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


net = resnet18()
optimized_net = paddlefx.optimize(my_compiler)(net)

x = paddle.rand([1, 3, 224, 224])
out = net(x)
res = optimized_net(x)

np.testing.assert_equal(res.numpy(), out.numpy())
