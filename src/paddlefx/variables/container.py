from __future__ import annotations

from .base import VariableBase


class ContainerVariable(VariableBase):
    def __init__(self, value: ContainerType):
        super().__init__()
        self.value = value

    def getitem(self, index):
        return self.value[index]

    def __str__(self):
        return str(self.value)


class TupleVariable(ContainerVariable):
    def __init__(self, value: tuple[VariableBase, ...]):
        super().__init__(value)

    def to_list(self):
        return ListVariable(list(self.value))


class ListVariable(ContainerVariable):
    def __init__(self, value: list[VariableBase]):
        super().__init__(value)


class DictVariable(ContainerVariable):
    pass


class RangeVariable(ContainerVariable):
    pass
