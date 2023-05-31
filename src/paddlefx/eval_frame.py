from __future__ import annotations

import dataclasses
import dis
import functools
import inspect
import types

from typing import Callable

import paddle
import paddle.nn

from ._eval_frame import set_eval_frame
from .bytecode_transformation import assemble, get_code_keys
from .translator import InstructionTranslator, convert_instruction


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


class DynamoContext:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        self.old_callback = set_eval_frame(self.callback)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        set_eval_frame(self.old_callback)

    def __call__(self, fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            old_callback = set_eval_frame(self.callback)

            # try:
            #     return fn(*args, **kwargs)
            # finally:
            #     set_eval_frame(old_callback)

            result = fn(*args, **kwargs)
            set_eval_frame(old_callback)
            return result

        return _fn


class DisableContext(DynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def disable(fn=None):
    return DisableContext()(fn)


def _compile(
    frame: types.FrameType,
    compiler_fn: Callable,
) -> GuardedCode:
    # TODO(zrr1999): This part can be removed when running the converted bytecode in the future.
    paddle_modules = [
        "paddle.nn",
        "paddle.fluid",
        "paddle.tensor",
        # TODO(zrr1999): add more modules
    ]
    module = inspect.getmodule(frame)
    if module is None:
        raise RuntimeError('Cannot find module for frame')
    package_name = module.__name__

    f_code = frame.f_code
    for paddle_module in paddle_modules:
        if package_name.startswith(paddle_module):
            return GuardedCode(f_code)
    instructions = list(map(convert_instruction, dis.get_instructions(f_code)))

    # tracer
    tracer = InstructionTranslator(
        instructions=instructions,
        frame=frame,
        compiler_fn=compiler_fn,
    )
    tracer.run()

    output = tracer.output
    instructions = output.output_instructions

    keys = get_code_keys()
    code_options = {k: getattr(f_code, k) for k in keys}
    bytecode = assemble(instructions)
    code_options["co_code"] = bytecode
    code = types.CodeType(*[code_options[k] for k in keys])

    # [print(x) for x in list(dis.get_instructions(code))]

    # debug, no trace
    return None

    g = GuardedCode(code)
    return g


def has_tensor_in_frame(frame: types.FrameType) -> bool:
    # NOTE: skip paddle internal code
    if frame.f_code.co_filename.endswith('paddle/fluid/dygraph/math_op_patch.py'):
        return False
    if frame.f_code.co_name == 'in_dygraph_mode':
        return False

    for v in frame.f_locals.values():
        # TODO: supprt containers
        if isinstance(v, paddle.Tensor):
            return True

    return False


def convert_frame_assert(compiler_fn: Callable):
    def _convert_frame_assert(frame: types.FrameType):
        if not has_tensor_in_frame(frame):
            return None

        return _compile(frame, compiler_fn)

    return _convert_frame_assert


def optimize(backend=None):
    def convert_frame(compiler_fn):
        inner_convert = convert_frame_assert(compiler_fn)

        def _convert_frame(frame: types.FrameType):
            result = inner_convert(frame)
            return result

        return _convert_frame

    return DynamoContext(convert_frame(backend))
