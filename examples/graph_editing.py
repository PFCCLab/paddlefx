from __future__ import annotations

import paddle

from paddlefx import symbolic_trace


def net(x, y):
    return x + y


traced_layer = symbolic_trace(net)
graph = traced_layer.graph

print("Before editing:")
print(traced_layer.get_source())

for node in graph.nodes:
    if node.op == 'call_function':
        with graph.inserting_after(node):
            new_node = graph.create_node(
                node.op, paddle.add, args=(node.args[0], node.args[0]), kwargs={}
            )
            node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        break

print("After editing:")
print(traced_layer.get_source())
