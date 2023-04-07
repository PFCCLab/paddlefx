from __future__ import annotations

from typing import Any

import paddle
import pydot

import paddlefx

from paddlefx.graph import _format_args, _qualified_name

# node op list
_NODE_OP_LIST = ["placeholder", "call_function", "call_module", "get_param", "output"]

# node op color map used in graphviz
_NODE_OP_COLOR_MAP = {
    "placeholder": "aliceblue",
    "call_function": "beige",
    "call_module": "cadetblue",
    "get_param": "darkgrey",
    "output": "darkturquoise",
}


class FxGraphViewer:
    """FxGraphViewer is used to visualize the paddlefx traced layer graph."""

    def __init__(self, traced_layer: paddlefx.GraphLayer, model_name: str):
        self.traced_layer = traced_layer
        self.model_name = model_name

    def _get_leaf_node(
        self, layer: paddle.nn.Layer, node: paddlefx.Node
    ) -> paddle.nn.Layer:
        py_obj = layer
        assert isinstance(node.target, str)
        atoms = node.target.split(".")
        for atom in atoms:
            if not hasattr(py_obj, atom):
                raise RuntimeError(
                    str(py_obj) + " does not have attribute " + atom + "!"
                )
            py_obj = getattr(py_obj, atom)
        return py_obj

    def _typename(self, target: Any) -> str:
        if isinstance(target, paddle.nn.Layer):
            ret = target.__class__.__name__
        elif isinstance(target, str):
            ret = target
        else:
            ret = _qualified_name(target)
        return ret

    def _get_node_label(
        self, traced_layer: paddlefx.GraphLayer, node: paddlefx.Node
    ) -> str:
        label = f"name={node.name}|op_code={node.op}\n"
        if node.op == "call_module":
            leaf_layer = self._get_leaf_node(traced_layer, node)
            label += f"\n{self._typename(leaf_layer)}\n"
            label += f"|target={self._typename(node.target)}\n"
            if node.args:
                label += f"|args=({_format_args(node.args, {})})\n"
            if node.kwargs:
                label += f"|kwargs=({_format_args([], node.kwargs)})\n"
            if node.users:
                label += f"|num_users={len(node.users)}\n"

        return "{" + label + "}"

    def _to_dot(self, traced_layer: paddlefx.GraphLayer, model_name: str) -> pydot.Dot:
        node_style = {
            "shape": "record",
            "fillcolor": "antiquewhite",
            "style": '"filled,rounded"',
        }

        dot = pydot.Dot(name=model_name, rankdir="TB")
        for node in traced_layer.graph.nodes:
            if node.op in _NODE_OP_LIST:
                label = self._get_node_label(traced_layer, node)
                node_style["fillcolor"] = _NODE_OP_COLOR_MAP[node.op]
                node_name = id(node)
                dot.add_node(pydot.Node(node_name, label=label, **node_style))
                for user in node.users:
                    user_name = id(user)
                    dot.add_edge(pydot.Edge(node_name, user_name))
            else:
                raise NotImplementedError(f'node: {node.op} {node.target}')
        return dot

    def get_graph_dot(self) -> pydot.Dot:
        return self._to_dot(self.traced_layer, self.model_name)
