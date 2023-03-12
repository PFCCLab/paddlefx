from __future__ import annotations

import dis
import types

from functools import partial
from typing import List

import paddlefx


def simple_callback(frame: types.FrameType, supported_ops: List[str] = []):
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in supported_ops:
        return None

    print(frame)
    print(dis.disassemble(frame.f_code))

    f_code = frame.f_code
    g = paddlefx.GuardedCode(f_code)
    return g


def add(a, b):
    print('\tcall add')
    c = a + b
    return c


# simple_callback
callback = partial(simple_callback, supported_ops=['add'])
with paddlefx.DynamoContext(callback):
    res = add(1, 3)
print(res)
assert res == 4
