from __future__ import annotations

from typing import TYPE_CHECKING

from .base import VariableBase

if TYPE_CHECKING:
    pass


class ConstantVariable(VariableBase):
    """ConstantVariable is a subclass of VariableBase used to wrap a Variable of the const type.

    Args:
        var(Any): The var to be wrapped.
    """

    def __str__(self):
        return str(self.var)

    def get_py_var(self, allow_tensor=False):
        return self.var
