import paddle

from paddlefx import symbolic_trace


def net(x, y):
    x = x * x
    x = x + y
    x = paddle.add(x=x, y=x)
    return paddle.nn.functional.relu(x=x)


traced_layer = symbolic_trace(net)

example_input_x = paddle.rand([3, 4])
example_input_y = paddle.rand([3, 4])

orig_output = net(example_input_x, example_input_y)
traced_output = traced_layer(example_input_x, example_input_y)

assert paddle.allclose(orig_output, traced_output)

print(f"python IR for {net.__name__}")
traced_layer.graph.print_tabular()
