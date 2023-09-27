from __future__ import annotations

import warnings

from typing import Any, Callable, Union

import paddle

BaseArgumentTypes = Union[str, int, float, bool, complex, paddle.dtype, paddle.Tensor]
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]


# Nodes represent a definition of a value in our graph of operators.
class Node:
    def __init__(self, graph, name, op, target, args, kwargs):
        self.graph = graph
        self.name = name  # unique name of value being created
        # TODO: whether call_module should be call_layer?
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|getattr
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or paddle.add

        # Currently, we do not support directly set the args and kwargs of a Node.
        # Please use `_update_args_kwargs` instead.
        self.args: tuple = ()
        self.kwargs = {}
        self._update_args_kwargs(args, kwargs)
        # Is a dict to act as an "ordered set". Keys are significant, value dont-care
        self.users: dict[Node, None] = {}

        self._prev = self
        self._next = self
        self._erased = False

    def __repr__(self):
        return self.name

    @property
    def next(self) -> Node:
        return self._next

    @property
    def prev(self) -> Node:
        return self._prev

    def prepend(self, x: Node) -> None:
        assert self.graph == x.graph, "Attempting to move a Node into a different Graph"
        if self == x:
            warnings.warn(
                "Trying to prepend a node to itself. This behavior has no effect on the graph."
            )
            return
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def append(self, x: Node) -> None:
        self._next.prepend(x)

    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p
        self._prev, self._next = self, self

    def _mark_uses(self, a, user):
        def add_use(n: Node):
            n.users.setdefault(user)
            return n

        map_arg(a, add_use)

    def _mark_unused(self, a, user):
        def remove_use(n: Node):
            if user in n.users:
                n.users.pop(user)
            return n

        map_arg(a, remove_use)

    def _update_args_kwargs(self, new_args, new_kwargs):
        # 1. Notify all the nodes that we used to use that we no longer use them
        self._mark_unused(self.args, self)
        self._mark_unused(self.kwargs, self)

        # 2. Update the args and kwargs
        self.args = new_args
        self.kwargs = new_kwargs

        # 3. Notify all the nodes that we now use that we use them
        self._mark_uses(self.args, self)
        self._mark_uses(self.kwargs, self)

    def replace_all_uses_with(self, replace_with: Node):
        to_process = list(self.users)
        for use_node in to_process:

            def maybe_replace_node(n: Node) -> Node:
                if n == self:
                    return replace_with
                else:
                    return n

            use_node._update_args_kwargs(
                map_arg(use_node.args, maybe_replace_node),
                map_arg(use_node.kwargs, maybe_replace_node),
            )


def map_aggregate(a, fn):
    """Apply fn to each Node appearing arg.

    arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, tuple):
        t = tuple(map_aggregate(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(a, '_fields') else type(a)(*t)
    elif isinstance(a, list):
        return list(map_aggregate(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return dict((k, map_aggregate(v, fn)) for k, v in a.items())
    elif isinstance(a, slice):
        return slice(
            map_aggregate(a.start, fn),
            map_aggregate(a.stop, fn),
            map_aggregate(a.step, fn),
        )
    else:
        return fn(a)


def map_arg(a: Any, fn: Callable[[Node], Any]) -> Any:
    """Apply fn to each Node appearing arg.

    arg may be a list, tuple, slice, or dict with string keys.
    """
    assert callable(fn), "fn must be a callable"
    return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)
