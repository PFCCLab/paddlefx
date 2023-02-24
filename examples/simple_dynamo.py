import dis
import types

import opcode  # noqa

from paddlefx._eval_frame import set_eval_frame

# a: types.CodeType = None
# a.co_code


def callback(frame: types.FrameType):
    print('enter callback')
    print(frame)
    # print(dis.disassemble(frame.f_code))
    f_code = frame.f_code
    return f_code


def add(a, b):
    print('call add')
    c = a + b
    return c


def func(a=1, b=3):
    print('call func')
    c = add(a, b)
    d = add(c, a)
    return d


print('set_eval_frame(callback)')
set_eval_frame(callback)

func(1, 3)
# add(1, 3)

print('\nset_eval_frame(None)')
set_eval_frame(None)

# func(1, 4)
