from __future__ import annotations

import operator

from typing import Any

from .dispatcher import Dispatcher
from .variables import ContainerVariable


@Dispatcher.register_decorator(operator.getitem)
def dispatcher_getitem(var: ContainerVariable, index: Any):
    return var.getitem(index)


Dispatcher.register(
    iter,
    ("VariableBase",),
    lambda variable: variable.get_iter(),
)
