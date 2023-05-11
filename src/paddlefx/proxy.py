from __future__ import annotations

import dis
import inspect
import operator
import typing

if typing.TYPE_CHECKING:
    from typing import Any

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
        return f'{self.node.name}'

    def __getattr__(self, attr: str):
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return Attribute(self, attr)

    def __getitem__(self, key: Any):
        # note:  If you don’t add the __getitem__ part, the type checking tool will report an error
        pass

    def __iter__(self):
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        instructions = list(dis.get_instructions(calling_frame.f_code))
        current_instruction_idx = calling_frame.f_lasti // 2
        current_instruction = instructions[current_instruction_idx]
        if current_instruction.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(current_instruction.argval))
        elif current_instruction.opname == "GET_ITER":
            return (self[i] for i in range(current_instruction.argval))
        raise ValueError("Cannot find UNPACK_SEQUENCE instruction")


class Attribute(Proxy):
    def __init__(self, root: Proxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getattr call
        if self._node is None:
            self._node = _create_proxy(
                self.tracer, 'call_function', getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __repr__(self):
        return f'{self.root}.{self.attr}'

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
            return _create_proxy(tracer, 'call_function', target, args, kwargs)

        impl.__name__ = method
        as_magic = f'__{method}__'
        setattr(Proxy, as_magic, impl)

    scope(method)
