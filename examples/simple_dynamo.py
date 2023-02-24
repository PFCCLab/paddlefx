import dis
import types

from paddlefx._eval_frame import set_eval_frame


def callback(frame: types.FrameType):
    print('enter callback')
    print(frame)
    print(dis.disassemble(frame.f_code))
    f_code = frame.f_code
    print()


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

print('\nset_eval_frame(None)')
set_eval_frame(None)

func(1, 4)
