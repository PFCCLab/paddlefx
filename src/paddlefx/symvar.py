from __future__ import annotations

import inspect
import itertools
import operator

from typing import TYPE_CHECKING, Any, Callable

from .source import LocalSource, Source

_sym_var_id_counter = itertools.count()

if TYPE_CHECKING:
    from .pyeval import PyEvalBase


class SymVar:
    def __init__(
        self,
        *,
        var: Any = None,
        vtype: Any = None,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ) -> None:
        self.var = var
        self.vtype = vtype if var is None else type(var)
        self.tx = tx
        self.source = source
        self.node = node

        self.id = f"id_{next(_sym_var_id_counter)}"

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        # TODO: just workaround, rm it later
        if self.source is not None and isinstance(self.source, LocalSource):
            return self.source.local_name
        elif self.node is not None:
            return self.node.name

        return f"SymVar({self.vtype}, {self.id})"

    def call(self, tx: PyEvalBase, *args, **kwargs) -> Any:
        # TODO: better org
        assert isinstance(self.var, Callable)
        var = self.var
        graph = tx.output.graph

        if var.__module__.startswith("paddle"):
            # TODO: support multiple ouputs and containers
            if 'nn.layer' in var.__module__:
                ot = args[0].vtype
                target = ''
                for name, layer in tx.f_locals['self']._sub_layers.items():
                    if var is layer:
                        target = name
                        break
                output = graph.call_module(target, args, kwargs)
                return SymVar(vtype=ot, node=output)
            else:
                ot = args[0].vtype
                output = graph.call_function(var, args, kwargs, ot)
                return SymVar(vtype=ot, node=output)
        elif inspect.isbuiltin(var):
            if var is print:
                raise NotImplementedError("print() is not supported")
            elif var is getattr:
                object, name = args
                attr = getattr(object.var, name.var)
                return SymVar(var=attr)
            elif var in [operator.add, operator.sub]:
                ot = args[0].vtype
                output = graph.call_function(var, args, kwargs, ot)
                return SymVar(vtype=ot, node=output)
            elif var in [operator.gt]:
                ot = args[0].vtype
                output = graph.call_function(var, args, kwargs, ot)
                return SymVar(vtype=ot, node=output)
            else:
                raise NotImplementedError(f"builtin {var} is not supported")

        return tx.inline_call_function(self, args, kwargs)
