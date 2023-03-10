import dataclasses
import dis
import types

from typing import Callable, List

from ._eval_frame import set_eval_frame
from .translator import InstructionTranslator, convert_instruction


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


def _compile(
    frame: types.FrameType,
    compiler_fn: Callable,
    supported_ops: List[str] = [],
):
    # NOTE: use supported_ops for frame skiping, eg: supported_ops = ['func', 'add']
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in supported_ops:
        return None

    code = frame.f_code
    instructions = list(map(convert_instruction, dis.get_instructions(code)))

    tracer = InstructionTranslator(instructions, frame, compiler_fn)
    tracer.run()

    # NOTE: just return the raw code from catched frame
    # TODO: support cache
    g = GuardedCode(code)
    return g


class DynamoContext:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        self.old_callback = set_eval_frame(self.callback)

    def __exit__(self, exc_type, exc_value, traceback):
        set_eval_frame(self.old_callback)

    def __call__(self, fn):
        def _fn(*args, **kwargs):
            old_callback = set_eval_frame(self.callback)

            result = fn(*args, **kwargs)

            set_eval_frame(old_callback)

            return result

        return _fn


def convert_frame_assert(compiler_fn: Callable):
    def _convert_frame_assert(frame: types.FrameType, supported_ops: List[str] = []):
        return _compile(frame, compiler_fn, supported_ops)

    return _convert_frame_assert


def optimize(backend=None, supported_ops: List[str] = []):
    def convert_frame(compiler_fn):
        inner_convert = convert_frame_assert(compiler_fn)

        def _convert_frame(frame: types.FrameType):
            result = inner_convert(frame, supported_ops)
            return result

        return _convert_frame

    return DynamoContext(convert_frame(backend))
