import paddle

from paddlefx import symbolic_trace


def net(x, y):
    x = x * x
    x = x + y
    x = paddle.add(x=x, y=x)
    return paddle.nn.functional.relu(x=x)


# tracing a paddle layer
graph = symbolic_trace(net)
raise
print("python IR:")
graph.print_tabular()
print("python code generated:")
src, _ = graph.python_code(root_module='self')
print(src)
