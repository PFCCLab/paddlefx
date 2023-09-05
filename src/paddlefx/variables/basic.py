from __future__ import annotations

from typing import Any

from .base import VariableBase


class ConstantVariable(VariableBase):
    """ConstantVariable is a subclass of VariableBase used to wrap a Variable of the const type.

    Args:
        value(Any): The value to be wrapped.
    """

    def __init__(
        self,
        value: Any,
    ):
        super().__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    def get_py_value(self, allow_tensor=False):
        return self.value
