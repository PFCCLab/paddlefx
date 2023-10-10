# type: ignore
from __future__ import annotations

import linecache

import paddle

from .graph import Graph

# normal exec loses the source code, however we can patch
# the linecache module to still recover it.
# using exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
_next_id = 0


def exec_with_source(src, globals):
    global _next_id
    key = f'<eval_with_key_{_next_id}>'
    _next_id += 1
    _eval_cache[key] = [line + '\n' for line in src.splitlines()]
    exec(compile(src, key, 'exec'), globals)


# patch linecache so that any code we exec using exec_with_source
# works with inspect
_eval_cache = {}
_orig_getlines = linecache.getlines


def patched_getline(*args, **kwargs):
    if args[0] in _eval_cache:
        return _eval_cache[args[0]]
    return _orig_getlines(*args, **kwargs)


linecache.getlines = patched_getline


class GraphLayer(paddle.nn.Layer):
    def __new__(cls, *args, **kwargs):
        # each instance of a graph module needs its own forward method
        # so create a new singleton class for each instance.
        # it is a subclass of the user-defined class, the only difference
        # is an extra layer to install the forward method
        class GraphLayerImpl(cls):
            pass

        return super().__new__(GraphLayerImpl)

    def __init__(self, root, graph: Graph):
        super().__init__()
        self.root = root
        if isinstance(root, paddle.nn.Layer):
            if hasattr(root, 'training'):
                self.training = root.training
            for node in graph.nodes:
                if node.op in ['get_attr', 'call_module']:
                    assert isinstance(node.target, str)
                    _copy_attr(root, self, node.target)
        else:
            raise RuntimeError('Unsupported type ' + str(root) + ' passed for root!')

        self.graph = graph
        self._generate_forward()

    def _generate_forward(self):
        body, free_variables = self.graph.python_code(root_module='self')
        if "self" not in free_variables:
            free_variables.insert(0, "self")
        body = '\n'.join('    ' + line for line in body.split('\n')) + '\n'
        self.src = f"""\
def forward({', '.join(free_variables)}):
    self = self.root
{body}
"""
        # install forward into the classes dictionary, this is what normally happens in the
        # 'class' statement
        # __new__ ensured that each instance has its own class
        gbls = {'paddle': paddle}
        exec_with_source(self.src, gbls)
        cls = type(self)
        for k, v in gbls.items():
            setattr(cls, k, v)

        code = self.forward.__code__
        self.forward = paddle.jit.to_static(self.forward)
        self.forward.__code__ = code

    def get_source(self, update: bool = True):
        if update:
            self._generate_forward()
        return self.src


# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def _copy_attr(from_module: paddle.nn.Layer, to_module: paddle.nn.Layer, target: str):
    *prefix, field = target.split('.')
    for item in prefix:
        f = getattr(from_module, item)
        t = getattr(to_module, item, None)
        if f is t:
            # we have already installed one of its parents
            # (e.g. target = root.linear.weight, but we have already installed root.linear)
            # once we install a parent, we no longer need to copy the children
            # since all the needed properties will already be present
            return

        if t is None:
            t = paddle.nn.Layer()
            setattr(to_module, item, t)
        from_module, to_module = f, t

    orig = getattr(from_module, field)
    # If it is a tensor and not a parameter attribute of a module, it should be a named buffer.
    # So, we register it as a named buffer in the target module.
    if isinstance(orig, paddle.Tensor) and not isinstance(orig, paddle.nn.Layer):
        to_module.register_buffer(field, orig)
    else:
        setattr(to_module, field, orig)
