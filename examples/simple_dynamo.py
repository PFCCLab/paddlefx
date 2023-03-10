import dis
import types

from typing import List

import paddle
import paddle.nn

import paddlefx


def my_compiler(gm: paddlefx.GraphLayer, example_inputs: List[paddle.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


def simple_callback(frame: types.FrameType):
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in ['func', 'add']:
        return None

    print(frame)
    print(dis.disassemble(frame.f_code))

    f_code = frame.f_code
    g = paddlefx.GuardedCode(f_code)
    return g


def add0(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(my_compiler)
def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(my_compiler)
def func(a=1, b=3):
    print('\tcall func')
    c = add(a, b)
    d = add(c, a)
    return d


# func(1, 3)
res = add(1, 3)
print(res)

with paddlefx.DynamoContext(simple_callback):
    res = add0(1, 3)
    print(res)
