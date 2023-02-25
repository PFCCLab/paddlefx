import dataclasses
import dis  # noqa
import types

import opcode  # noqa

from paddlefx._eval_frame import set_eval_frame


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


def callback(frame: types.FrameType):
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in ['func', 'add']:
        return None

    print('enter callback')
    # print(frame)
    # print(dis.disassemble(frame.f_code))
    f_code = frame.f_code
    g = GuardedCode(f_code)
    return g
    # return None


def add(a, b):
    print('\tcall add')
    c = a + b
    return c


def func(a=1, b=3):
    print('\tcall func')
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
