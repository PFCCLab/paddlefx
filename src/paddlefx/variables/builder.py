from __future__ import annotations

import dataclasses

from typing import Any


@dataclasses.dataclass
class GraphArg:
    example: Any
    is_tensor: bool = True

    def get_examples(self):
        return [self.example]

    def __len__(self):
        return 1
