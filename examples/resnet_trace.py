import paddle
import paddle.nn

from paddle.vision.models import resnet18

from paddlefx import symbolic_trace

net = resnet18()

# tracing a paddle layer
graph = symbolic_trace(net)

print("python IR:")
graph.print_tabular()
print("python code generated:")
src, _ = graph.python_code(root_module='self')
print(src)

# TODO: need to implement GraphModule (or GraphLayer to align with paddle.nn.Layer),
# then, validate the traced net is correct by
# comparing the output of traced net and original net
# x = paddle.rand([1, 3, 224, 224])
