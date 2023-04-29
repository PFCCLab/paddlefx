from __future__ import annotations

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
    if hasattr(func, '__name__'):
        name = func.__name__
    else:
        name = func.__class__.__name__

    # things like getattr just appear in builtins
    if getattr(builtins, name, None) is func:
        return name
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
    if hasattr(orig_method, '__name__'):
        name = orig_method.__name__
    else:
        name = orig_method.__class__.__name__
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


class _InsertPoint:
    def __init__(self, graph, new_insert):
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        self.graph._insert = self.orig_insert


class _node_list:
    def __init__(self, graph: Graph, direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __len__(self):
        return self.graph._len

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            if not cur._erased:
                yield cur
            cur = getattr(cur, direction)

    def __reversed__(self):
        return _node_list(self.graph, '_next' if self.direction == '_prev' else '_prev')


class Graph:
    def __init__(self):
        self._used_names = {}  # base name -> number
        self._root = Node(self, '', 'root', '', (), {})
        self._len = 0
        # Set the default insert point to the graph trailing
        self._insert = self._root.prepend

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
        self._insert(n)
        self._len += 1
        return n

    def output(self, result):
        return self.create_node(op='output', target='output', args=(result,))

    def _name(self, op):
        if hasattr(op, '__name__'):
            op = op.__name__
        elif hasattr(op, '__class__'):
            op = op.__class__.__name__

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

    def erase_node(self, to_erase: Node) -> None:
        if len(to_erase.users) > 0:
            raise RuntimeError(
                f'Tried to erase Node {to_erase} but it still had {len(to_erase.users)} '
                f'users in the graph: {to_erase.users}!'
            )

        to_erase._remove_from_list()
        to_erase._erased = True  # iterators may retain handles to erased nodes
        self._len -= 1

        # Null out this Node's argument nodes so that the Nodes referred to
        # can update their ``users`` accordingly
        to_erase._update_args_kwargs(
            map_arg(to_erase.args, lambda n: None),
            map_arg(to_erase.kwargs, lambda n: None),
        )

    def inserting_before(self, n: Node | None = None):
        if n is None:
            return self.inserting_after(self._root)
        assert n.graph == self, "Node to insert before is not in graph."
        return _InsertPoint(self, n.prepend)

    def inserting_after(self, n: Node | None = None):
        if n is None:
            return self.inserting_before(self._root)
        assert n.graph == self, "Node to insert after is not in graph."
        return _InsertPoint(self, n.append)

    @property
    def nodes(self):
        return _node_list(self)

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
                body.append(f'return {node.args[0]}\n')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        src = ''.join(body)
        return src, free_vars

    def print_tabular(self, print_mode="tabulate"):
        """Prints the intermediate representation of the graph in tabular
        format.

        Note that this API allows users to choose between using the ``raw``,
        ``tabulate`` or ``rich`` mode. If the user specifies a mode that is not
        installed, the API will automatically fall back on the ``raw`` mode.
        """
        assert print_mode in ["raw", "tabulate", "rich"]
        if print_mode == "raw":
            node_specs = [
                " ".join(
                    map(str, [v for v in [n.op, n.name, n.target, n.args, n.kwargs]])
                )
                for n in self.nodes
            ]
            print(" ".join(['opcode', 'name', 'target', 'args', 'kwargs']))
            print("\n".join(node_specs))
        elif print_mode == "tabulate":
            try:
                from tabulate import tabulate

                node_specs = [
                    [n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes
                ]
                print(
                    tabulate(
                        node_specs,
                        headers=['opcode', 'name', 'target', 'args', 'kwargs'],
                    )
                )
            except ImportError:
                import warnings

                warnings.warn(
                    "`print_tabular` relies on the library `tabulate`, "
                    "which could not be found on this machine. Run `pip "
                    "install tabulate` to install the library."
                )
                self.print_tabular("raw")
        elif print_mode == "rich":
            try:
                import rich
                import rich.table

                table = rich.table.Table(
                    'opcode', 'name', 'target', 'args', 'kwargs', expand=True
                )
                for n in self.nodes:
                    table.add_row(
                        *map(
                            str, [v for v in [n.op, n.name, n.target, n.args, n.kwargs]]
                        )
                    )
                rich.print(table)
            except ImportError:
                import warnings

                warnings.warn(
                    "`print_tabular` relies on the library `rich`, "
                    "which could not be found on this machine. Run `pip "
                    "install rich` to install the library."
                )
                self.print_tabular("raw")
