from __future__ import annotations

import operator
import typing

from typing import Any, Callable, Dict, Iterable, Iterator, Optional, OrderedDict, Tuple

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

    def __getattr__(self, k):
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, k)


class Attribute(Proxy):
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = _create_proxy(
                self.tracer, 'call_function', getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return _create_proxy(
            self.tracer, 'call_method', self.attr, (self.root,) + args, kwargs
        )


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
