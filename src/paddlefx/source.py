from __future__ import annotations

import dataclasses


class Source:
    def name(self) -> str:
        raise NotImplementedError()

    def is_traceable(self) -> bool:
        raise NotImplementedError()

    def need_guard(self) -> bool:
        # TODO(zrr1999): implement is_traceable
        return True


@dataclasses.dataclass(frozen=True)
class LocalSource(Source):
    local_name: str

    def name(self):
        return f"L[{repr(self.local_name)}]"


@dataclasses.dataclass(frozen=True)
class GlobalSource(Source):
    global_name: str

    def name(self):
        return f"G[{repr(self.global_name)}]"
