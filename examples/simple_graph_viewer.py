from __future__ import annotations

from paddle.vision.models import resnet18

from paddlefx import FxGraphViewer, symbolic_trace

net = resnet18()
traced_layer = symbolic_trace(net)

g = FxGraphViewer(traced_layer, "resnet18")
g.get_graph_dot().write_svg("resnet.svg")
