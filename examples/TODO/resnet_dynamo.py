from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

from paddle.vision.models import resnet18

import paddlefx

from paddlefx.compiler.tvm import TVMCompiler

compiler = TVMCompiler(full_graph=True, print_tabular_mode="rich")
net = resnet18()
optimized_net = paddlefx.optimize(net, backend=compiler)

x = paddle.rand([1, 3, 224, 224])
out = net(x)
res = optimized_net(x)

np.testing.assert_equal(res.numpy(), out.numpy())
