import paddle
from paddlefx import symbolic_trace

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._fc1 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc2 = paddle.nn.Linear(in_features=10, out_features=10)
        self._fc3 = paddle.nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        x = self._fc1(x)
        x = self._fc2(x)
        x = self._fc3(x)
        y = paddle.add(x=x, y=x)
        return paddle.nn.functional.relu(x=y)

net = MyNet()

# tracing a paddle layer
graph = symbolic_trace(net)

print("python IR:")
graph.print_tabular()
print("python code generated:")
src, _ = graph.python_code(root_module='self')
print(src)
