import dis
import types

from functools import partial
from typing import List

import paddle
import paddle.nn

import paddlefx


def my_compiler(gm: paddlefx.GraphLayer, example_inputs: List[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


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


@paddlefx.optimize(my_compiler, supported_ops=['add', 'func'])
def func(a=1, b=3):
    print('\tcall func')
    c = add(a, b)
    d = add(c, a)
    return d


# paddlefx.optimize
res = func(1, 3)
print(res)

# simple_callback
callback = partial(simple_callback, supported_ops=['add'])
with paddlefx.DynamoContext(callback):
    res = add(1, 3)
print(res)
