from __future__ import annotations

import dis
import types

from functools import partial

import paddlefx


def simple_callback(frame: types.FrameType):
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
callback = partial(simple_callback)
with paddlefx.DynamoContext(callback):
    res = add(1, 3)
print(res)
assert res == 4
