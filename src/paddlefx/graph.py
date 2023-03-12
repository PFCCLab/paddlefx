import builtins
import keyword

from typing import Any

import paddle
import paddle.nn

from .node import Node
from .proxy import magic_methods


def _is_magic(x):
    return x.startswith('__') and x.endswith('__')


def snake_case(s):
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')


def _qualified_name(func):
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    return f'{module}.{name}'


def _is_illegal_name(name: str, obj: Any) -> bool:
    # 1. keywords are never allowed as names.
    if name in keyword.kwlist:
        return True

    # 2. Can't shadow a builtin name, unless you *are* that builtin.
    if name in builtins.__dict__:
        return obj is not builtins.__dict__[name]

    return False


def _find_module_of_method(orig_method):
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [paddle, paddle.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')


def _format_args(args, kwargs):
    args_s = ', '.join(repr(a) for a in args)
    kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f'{args_s}, {kwargs_s}'
    return args_s or kwargs_s


def _format_target(base, target):
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r


def map_arg(a, fn):
    """apply fn to each Node appearing arg.

    arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, (tuple, list)):
        return type(a)(map_arg(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return {k: map_arg(v, fn) for k, v in a.items()}
    elif isinstance(a, slice):
        return slice(map_arg(a.start, fn), map_arg(a.stop, fn), map_arg(a.step, fn))
    elif isinstance(a, Node):
        return fn(a)
    else:
        return a


class Graph:
    def __init__(self):
        self.nodes = []
        self._used_names = {}  # base name -> number

    def _mark_uses(self, a):
        def add_use(n: Node):
            n.uses += 1
            return n

        map_arg(a, add_use)

    def create_node(self, op, target=None, args=None, kwargs=None, name=None):
        assert op in (
            'call_function',
            'call_method',
            'get_param',
            'call_module',
            'placeholder',
            'output',
        )
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        self._mark_uses(args)
        self._mark_uses(kwargs)
        name = name if name is not None else self._name(target or op)
        if name[0].isdigit():
            name = f'_{name}'
        n = Node(
            self,
            name,
            op,
            target,
            args,
            kwargs,
        )
        self.nodes.append(n)
        return n

    def output(self, result):
        return self.create_node(op='output', target='output', args=result)

    def _name(self, op):
        if hasattr(op, '__name__'):
            op = op.__name__

        if _is_magic(op):
            op = op[2:-2]
        op = op.replace('.', '_')
        op = snake_case(op)

        if op not in self._used_names:
            self._used_names[op] = 0
            if (
                not hasattr(paddle, op)
                and not hasattr(paddle.nn.functional, op)
                and not hasattr(paddle.nn, op)
                and not _is_illegal_name(op, None)
            ):
                return op
        i = self._used_names[op] = self._used_names[op] + 1
        return f'{op}_{i}'

    def get_param(self, target):
        return self.create_node('get_param', target)

    def placeholder(self, name):
        return self.create_node('placeholder', target=name, name=name.replace('*', ''))

    def python_code(self, root_module):
        free_vars = []
        body = []
        for node in self.nodes:
            if node.op == 'placeholder':
                free_vars.append(node.target)
                if node.target != node.name:
                    body.append(f'{node.name} = {node.target}\n')
                continue
            elif node.op == 'call_method':
                body.append(
                    f'{node.name} = {_format_target(repr(node.args[0]), node.target)}'
                    f'({_format_args(node.args[1:], node.kwargs)})\n'
                )
                continue
            elif node.op == 'call_function':
                # pretty print operators
                if (
                    node.target.__module__ == '_operator'
                    and node.target.__name__ in magic_methods
                ):
                    body.append(
                        f'{node.name} = {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}\n'
                    )
                    continue
                qualified_name = _qualified_name(node.target)
                if (
                    qualified_name == 'getattr'
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                ):
                    # pretty print attribute access
                    body.append(
                        f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}\n'
                    )
                    continue
                body.append(
                    f'{node.name} = {qualified_name}({_format_args(node.args, node.kwargs)})\n'
                )
                continue
            elif node.op == 'call_module':
                body.append(
                    f'{node.name} = {_format_target(root_module,node.target)}({_format_args(node.args, node.kwargs)})\n'
                )
                continue
            elif node.op == 'get_param':
                body.append(
                    f'{node.name} = {_format_target(root_module, node.target)}\n'
                )
                continue
            elif node.op == 'output':
                body.append(f'return {node.args}\n')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        src = ''.join(body)
        return src, free_vars

    def print_tabular(self):
        """Prints the intermediate representation of the graph in tabular
        format.

        Note that this API requires the ``tabulate`` module to be installed.
        """
        try:
            from tabulate import tabulate
        except ImportError:
            print(
                "`print_tabular` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library."
            )
        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes]
        print(
            tabulate(node_specs, headers=['opcode', 'name', 'target', 'args', 'kwargs'])
        )
