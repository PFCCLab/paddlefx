from __future__ import annotations

import itertools

from typing import TYPE_CHECKING, Any

from ..proxy import Proxy
from ..source import LocalSource, Source

_sym_var_id_counter = itertools.count()

if TYPE_CHECKING:
    from ..pyeval import PyEvalBase


class VariableBase:
    def __init__(
        self,
        *,
        var: Any = None,
        vtype: Any = None,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ) -> None:
        # TODO: remove var and vtype
        self.var = var
        self.vtype = vtype if var is None else type(var)
        self.tx = tx
        self.source = source
        self.node = node

        self.id = f"id_{next(_sym_var_id_counter)}"

    def __str__(self) -> str:
        # TODO: just workaround, rm it later
        if self.source is not None and isinstance(self.source, LocalSource):
            return self.source.local_name
        elif self.node is not None:
            return self.node.name

        return f"VariableBase({self.vtype}, {self.id})"

    def __repr__(self) -> str:
        if self.source is not None and isinstance(self.source, LocalSource):
            return self.source.local_name
        elif self.node is not None:
            return self.node.name
        # specificed for the fixed variable
        elif self.var is not None:
            return str(self.var)

        return f"VariableBase({self.vtype}, {self.id})"


class ObjectVariable(VariableBase):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def __str__(self):
        return str(self.obj)

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        return translator.output.create_node(
            "call_method", "__call__", [self] + args, kwargs
        )

    def call_method(
        self,
        translator,
        name: str,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> ObjectVariable:
        # proxy_args, proxy_kwargs = proxy_args_kwargs([self] + args, kwargs)
        # return ObjectVariable(
        #     translator.output.create_proxy(
        #         "call_method", name, proxy_args, proxy_kwargs
        #     )
        # )
        return ObjectVariable(
            translator.output.create_node("call_method", name, args, kwargs)
        )


# class BuiltinVariable(CallableVariable):
#     def call_print(
#         self,
#         translator,
#         *args: tuple[VariableBase],
#         **kwargs: dict[str, VariableBase],
#     ) -> ConstantVariable:
#         return ConstantVariable(None)

#     def call_getattr(
#         self, translator, obj: ObjectVariable, name: str
#     ) -> ObjectVariable:
#         return ObjectVariable(
#             translator.output.create_node("call_method", "__getattr__", [obj, name])
#         )


class LayerVariable(ObjectVariable):
    def __init__(self, target: str):
        super().__init__(target)
        # TODO: those are used to generate code
        self.args = []
        self.kwargs = {}

    def __str__(self):
        args = ", ".join(self.args)
        kwargs = ", ".join(self.kwargs)
        return f"{self.obj}({args}, {kwargs})"

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        self.args = [str(a) for a in args]
        self.kwargs = [f"k{k}={v}" for k, v in kwargs.items()]
        return translator.output.create_node("call_module", self.obj, args, kwargs)


class TensorVariable(ObjectVariable):
    def __init__(self, proxy: Proxy):
        super().__init__(proxy)

    def as_proxy(self) -> Proxy:
        return self.obj
