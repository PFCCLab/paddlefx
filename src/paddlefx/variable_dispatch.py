from __future__ import annotations

from .dispatcher import Dispatcher

# iter
Dispatcher.register(
    iter,
    ("VariableBase",),
    lambda variable: variable.get_iter(),
)
