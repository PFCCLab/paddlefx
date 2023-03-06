# type: ignore
import linecache

import paddle

from paddlefx import Graph

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
        self.graph = graph
        self._generate_forward()

    def _generate_forward(self):
        body, free_variables = self.graph.python_code(root_module='self')
        body = '\n'.join('    ' + line for line in body.split('\n')) + '\n'
        self.src = f"""\
def forward(self, {', '.join(free_variables)}):
    self = self.root
{body}
"""
        # print(self.src)
        # install forward into the classes dictionary, this is what normally happens in the
        # 'class' statement
        # __new__ ensured that each instance has its own class
        gbls = {'paddle': paddle}
        exec_with_source(self.src, gbls)
        cls = type(self)
        for k, v in gbls.items():
            setattr(cls, k, v)
