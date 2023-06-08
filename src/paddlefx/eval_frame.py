from __future__ import annotations

import dataclasses
import dis
import functools
import logging
import types

from typing import Callable

import paddle
import paddle.nn

from ._eval_frame import set_eval_frame
from .bytecode_transformation import transform_code_object
from .translator import InstructionTranslator


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

            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(old_callback)

            # debug friendly
            # result = fn(*args, **kwargs)
            # set_eval_frame(old_callback)
            # return result

        # compiled fn
        _fn.fn = fn

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
    f_code = frame.f_code

    def transform(instructions, code_options):
        # tracer
        tracer = InstructionTranslator(
            instructions=instructions,
            frame=frame,
            code_options=code_options,
            compiler_fn=compiler_fn,
        )
        tracer.run()

        instructions[:] = tracer.output.output_instructions

    out_code = transform_code_object(f_code, transform)

    logging.debug(f"\nraw_code:")
    [logging.debug(x) for x in list(dis.get_instructions(f_code))]
    logging.debug(f"")

    logging.debug(f"\ntransformed_code:")
    [logging.debug(x) for x in list(dis.get_instructions(out_code))]
    logging.debug(f"")

    # debug, no trace
    # return None

    g = GuardedCode(out_code)
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
            logging.debug(f"frame skipped: {frame.f_code.co_name}")
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
