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


class CodeCacheManager:
    cache_dict: dict[types.CodeType, GuardedCodes] = {}

    @classmethod
    def add_cache(cls, code: types.CodeType, guarded_code: GuardedCode):
        cls.cache_dict.setdefault(code, [])
        cls.cache_dict[code].append(guarded_code)

    @classmethod
    def get_cache(cls, frame: types.FrameType) -> GuardedCode | None:
        code: types.CodeType = frame.f_code
        if code not in cls.cache_dict:
            print(f"Firstly call {code}\n")
            return None
        return cls.lookup(frame, cls.cache_dict[code])

    @classmethod
    def clear_cache(cls):
        cls.cache_dict.clear()

    @classmethod
    def lookup(
        cls, frame: types.FrameType, guarded_codes: GuardedCodes
    ) -> GuardedCode | None:
        for guarded_code in guarded_codes:
            try:
                guard_fn = guarded_code.guard_fn
                if guard_fn(frame):
                    print(
                        f"[Cache]: Cache hit, GuardFunction is {guard_fn}\n",
                    )
                    return guarded_code
                else:
                    print(
                        f"[Cache]: Cache miss, GuardFunction is {guard_fn}\n",
                    )
            except Exception as e:
                print(f"[Cache]: GuardFunction function error: {e}\n")
                continue

        print("[Cache]: all guards missed\n")
        return None
