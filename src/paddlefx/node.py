from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

import paddle

BaseArgumentTypes = Union[str, int, float, bool, complex, paddle.dtype, paddle.Tensor]
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]


# Nodes represent a definition of a value in our graph of operators.
class Node:
    def __init__(self, graph, name, op, target, args, kwargs):
        self.graph = graph
        self.name = name  # unique name of value being created
        self.op = op  # the kind of operation = placeholder|call_method|call_module|call_function|getattr
        self.target = target  # for method/module/function, the name of the method/module/function/attr
        # being invoked, e.g add, layer1, or paddle.add
        self.args = args
        self.kwargs = kwargs
        self.uses = 0

    def __repr__(self):
        return self.name


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
