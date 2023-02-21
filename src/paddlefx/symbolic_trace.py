import paddle

from .graph import Graph
from .node import Node
from .proxy import Proxy, _create_proxy


# in pytorch, it's find a module
# in paddle, it's find a layer
def _find_module(root, m):
    for n, p in root.named_children():
        if m is p:
            return n
    raise NameError('module is not installed as a submodule')


class Tracer:
    def __init__(self):
        self.graph = Graph()

    # For now, we only monkey patch addle.add, paddle.nn.functional.relu
    # TODO: We'll need a solution to patch all related paddle functions.
    def _monkey_patch_paddle_functions(self):
        # monkey patch paddle.add to create a proxy for it
        orig_add_call = paddle.add

        def paddle_add_wrapper(*args, **kwargs):
            return _create_proxy(
                self, 'call_function', orig_add_call, args, kwargs, 'add'
            )

        # monkey patch paddle.nn.functional.relu to create a proxy for it
        orig_relu_call = paddle.nn.functional.relu

        def paddle_relu_wrapper(*args, **kwargs):
            return _create_proxy(
                self, 'call_function', orig_relu_call, args, kwargs, 'relu'
            )

        paddle.add = paddle_add_wrapper
        paddle.nn.functional.relu = paddle_relu_wrapper

        return orig_add_call, orig_relu_call

    def _release_paddle_functions(self, orig_add_call, orig_relu_call):
        paddle.add = orig_add_call
        paddle.nn.functional.relu = orig_relu_call

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
            target = _find_module(root, mod)
            ### change it to create proxy in proxy.py
            return _create_proxy(self, 'call_module', target, args, kwargs, target)

        try:
            orig_add_call, orig_relu_call = self._monkey_patch_paddle_functions()
            paddle.nn.Layer.__call__ = module_call_wrapper
            self.graph.output(self.create_arg(fn(*args)))
        finally:
            self._release_paddle_functions(orig_add_call, orig_relu_call)
            paddle.nn.Layer.__call__ = orig_module_call

        return self.graph

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
        raise NotImplementedError(f"argument of type: {type(a)}")


def symbolic_trace(root):
    tracer = Tracer()
    return tracer.trace(root)
