from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..source import Source
from .base import VariableBase

if TYPE_CHECKING:
    from ..pyeval import PyEvalBase


class ContainerVariable(VariableBase):
    def __init__(
        self,
        container: Any,
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(var=container, tx=tx, source=source, node=node)

    def getitem(self, index):
        return self.var[index]


class TupleVariable(ContainerVariable):
    def __init__(
        self,
        tuple_value: tuple[VariableBase, ...],
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(tuple_value, tx=tx, source=source, node=node)

    def to_list(self):
        return ListVariable(list(self.var))


class ListVariable(ContainerVariable):
    def __init__(
        self,
        list_value: list[VariableBase],
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(list_value, tx=tx, source=source, node=node)


class DictVariable(ContainerVariable):
    pass


class RangeVariable(ContainerVariable):
    pass
