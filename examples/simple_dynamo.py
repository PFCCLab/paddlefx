import dis
import types

from paddlefx._eval_frame import set_eval_frame


def callback(frame: types.FrameType):
    print('enter callback')
    print(frame)
    print(dis.disassemble(frame.f_code))


def add(a, b):
    print('call add')
    return a + b


print('set_eval_frame(callback)')
set_eval_frame(callback)

add(1, 3)

print('\nset_eval_frame(None)')
set_eval_frame(None)

add(1, 4)
