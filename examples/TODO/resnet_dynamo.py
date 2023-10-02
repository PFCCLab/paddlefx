from __future__ import annotations

import numpy as np
import paddle
import paddle.nn
import paddle.tensor

from paddle.vision.models import resnet18

import paddlefx

from paddlefx.compiler.tvm import TVMCompiler

paddle.seed(1234)

# logging.getLogger().setLevel(logging.DEBUG)

compiler = TVMCompiler(full_graph=True, print_tabular_mode="rich")
net = resnet18(pretrained=True, num_classes=2)
optimized_net = paddlefx.optimize(net, backend=compiler)

x = paddle.rand([1, 3, 224, 224], dtype="float32")
out = net(x)
res = optimized_net(x)
np.testing.assert_allclose(res.numpy(), out.numpy(), rtol=1e-5, atol=1e-6)

for _ in range(10):
    out = net(x)

for _ in range(10):
    res = optimized_net(x)
