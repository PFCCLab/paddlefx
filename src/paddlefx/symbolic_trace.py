import builtins
import functools

from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import paddle
import paddle.nn

from .graph import Graph
from .graph_layer import GraphLayer
from .node import Node, base_types, map_aggregate
from .proxy import Proxy, _create_proxy

MODULES_TO_PATCH = (paddle, paddle.nn, paddle.nn.functional)
LAYERS_EXCLUDE_TO_PATCH = (paddle.nn.Sequential,)


# in pytorch, it's find a module
# in paddle, it's find a layer
def _find_module(root, m):
    # BFS search the whole tree to find the submodule
    children_queue = list(root.named_children())
    while children_queue:
        n, p = children_queue.pop(0)
        if m is p:
            return n
        children_queue.extend((f"{n}.{k}", v) for k, v in p.named_children())
    raise NameError('module is not installed as a submodule')


def _is_leaf_module(m) -> bool:
    return m.__module__.startswith("paddle.nn") and not isinstance(
        m, paddle.nn.Sequential
    )


class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError()


class _PatchedFnSetItem(_PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class _PatchedFnDel(_PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]


class _PatchedFnSetAttr(_PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class _Patcher:
    def __init__(self):
        super().__init__()
        self.patches_made = []
        self.visited = set()

    def patch(
        self,
        frame_dict,
        name,
        new_fn,
        deduplicate: bool = True,
    ):
        """Replace frame_dict[name] with new_fn until we exit the context manager."""
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        if name not in frame_dict and hasattr(builtins, name):
            self.patches_made.append(_PatchedFnDel(frame_dict, name, None))
        elif getattr(frame_dict[name], "__fx_already_patched", False):
            return  # already patched, no need to do it again
        else:
            self.patches_made.append(
                _PatchedFnSetItem(frame_dict, name, frame_dict[name])
            )
        frame_dict[name] = new_fn

    def patch_method(
        self, cls: type, name: str, new_fn: Callable, deduplicate: bool = True
    ):
        """Replace object_or_dict.name with new_fn until we exit the context manager."""
        new_fn.__fx_already_patched = deduplicate  # type: ignore[attr-defined]
        orig_fn = getattr(cls, name)
        if getattr(orig_fn, "__fx_already_patched", False):
            return  # already patched, no need to do it again
        self.patches_made.append(_PatchedFnSetAttr(cls, name, orig_fn))
        setattr(cls, name, new_fn)

    def visit_once(self, thing: Any):
        """Return True on the first call to with thing, otherwise false."""
        idx = id(thing)
        if idx in self.visited:
            return False
        self.visited.add(idx)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Undo all the changes made via self.patch() and self.patch_method()"""
        while self.patches_made:
            # unpatch in reverse order to handle duplicates correctly
            self.patches_made.pop().revert()
        self.visited.clear()


def _find_proxy(*objects_to_search):
    """Recursively search a data structure for a Proxy() and return it,
    return None if not found."""
    proxy = None

    def find_proxy(x):
        nonlocal proxy
        if isinstance(x, Proxy):
            proxy = x

    # find the first proxy in the args/kwargs
    map_aggregate(objects_to_search, find_proxy)
    return proxy


def _create_wrapped_func(orig_fn):
    @functools.wraps(orig_fn)
    def wrapped(*args, **kwargs):
        """Given an closed-over ``orig_function`` to invoke, search the args and kwargs for
        a Proxy object.

        If there is one, emit a ``call_function`` node to preserve the call to this
        leaf function directly. Otherwise, just return the results of this function
        call, as this function is not being traced.
        """
        proxy = _find_proxy(args, kwargs)
        if proxy is not None:
            return_proxy = _create_proxy(
                proxy.tracer, 'call_function', orig_fn, args, kwargs, orig_fn.__name__
            )
            return return_proxy
        return orig_fn(*args, **kwargs)

    return wrapped


def _autowrap_check(patcher: _Patcher, frame_dict: Dict[str, Any]):
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if (
                not name.startswith("_")
                and callable(value)
                and value not in LAYERS_EXCLUDE_TO_PATCH
            ):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))


class Tracer:
    def __init__(self):
        self.graph = Graph()

    def trace(self, root) -> Graph:
        is_layer = isinstance(root, paddle.nn.Layer)
        if is_layer:
            fn = type(root).forward
        else:
            fn = root

        # for now, it only support postional args
        # TODO: we should support keyword args
        skip_arg_idx = 0
        co = fn.__code__
        total_args = co.co_argcount
        names_iter = iter(co.co_varnames)
        args = []
        if is_layer:
            # Fill the first argument with self
            # and skips creating a proxy for self
            args.append(root)
            next(names_iter)
            skip_arg_idx = 1
        else:
            root = paddle.nn.Layer()

        # Fill other arguments
        for _ in range(skip_arg_idx, total_args):
            name = next(names_iter)
            args.append(self._proxy_placeholder(name))

        assert len(args) == total_args

        # monkey patch paddle.nn.Layer to create a proxy for it
        orig_module_call = paddle.nn.Layer.__call__

        def module_call_wrapper(mod, *args, **kwargs):
            if not _is_leaf_module(mod):
                # Run original __call__ to trace the submodules
                return orig_module_call(mod, *args, **kwargs)
            target = _find_module(root, mod)
            name = target.replace('.', '_')
            ### change it to create proxy in proxy.py
            return _create_proxy(self, 'call_module', target, args, kwargs, name)

        with _Patcher() as patcher:
            # step1: patch layer call
            patcher.patch_method(
                paddle.nn.Layer, "__call__", module_call_wrapper, deduplicate=False
            )
            # step2: patch paddle functions
            for module in MODULES_TO_PATCH:
                _autowrap_check(patcher, module.__dict__)
            # step3: trace it!
            self.graph.output(self.create_arg(fn(*args)))

        return GraphLayer(root, self.graph)

    def _proxy_placeholder(self, name):
        n = self.graph.create_node('placeholder', name, (), {})
        return Proxy(n, self)

    def create_node(self, op, target, args, kwargs, name=None):
        return self.graph.create_node(op, target, args, kwargs, name)

    def create_arg(self, a):
        if isinstance(a, (tuple, list)):
            return type(a)(self.create_arg(elem) for elem in a)
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                if not isinstance(k, str):
                    raise NotImplementedError(f"dictionaries with non-string keys: {a}")
                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(
                self.create_arg(a.start),
                self.create_arg(a.stop),
                self.create_arg(a.step),
            )

        if isinstance(a, Proxy):
            # base case: we unwrap the Proxy object
            return a.node
        elif isinstance(a, base_types) or a is None or a is ...:
            return a
        raise NotImplementedError(f"argument of type: {type(a)}")


def symbolic_trace(root):
    tracer = Tracer()
    return tracer.trace(root)
