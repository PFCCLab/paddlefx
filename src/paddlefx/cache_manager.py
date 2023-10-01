from __future__ import annotations

import dataclasses
import types

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    GuardFunction = Callable[[types.FrameType], bool]
    GuardedCodes = list["GuardedCode"]


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    guard_fn: GuardFunction
