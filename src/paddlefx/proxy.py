from __future__ import annotations

import operator
import typing

if typing.TYPE_CHECKING:
    from .node import Node
    from .symbolic_trace import Tracer


# Unwrap the proxies inside args, and kwargs, create the resulting node
# and then wrap the result in a proxy.
def _create_proxy(tracer: Tracer, op, target, args_, kwargs_, name=None):
    args = tracer.create_arg(args_)
    kwargs = tracer.create_arg(kwargs_)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    rn = tracer.create_node(op, target, args, kwargs, name)
    return Proxy(rn, tracer)


class Proxy:
    def __init__(self, node: Node, tracer: Tracer):
        self.node = node
        self.tracer = tracer

    def __repr__(self):
        return f'Proxy({self.node.name})'


reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and': '{} & {}',
    'or': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]',
}

magic_methods = dict(
    {
        'eq': '{} == {}',
        'ne': '{} != {}',
        'lt': '{} < {}',
        'gt': '{} > {}',
        'le': '{} <= {}',
        'ge': '{} >= {}',
        'pos': '+{}',
        'neg': '-{}',
        'invert': '~{}',
    },
    **reflectable_magic_methods,
)

for method in magic_methods:

    def scope(method):
        def impl(*args, **kwargs):
            tracer = args[0].tracer
            target = getattr(operator, method)
            return _create_proxy(tracer, 'call_function', target, args, kwargs, method)

        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Proxy, as_magic, impl)

    scope(method)
