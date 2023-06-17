from __future__ import annotations

import dataclasses
import dis
import logging
import types

from typing import Callable

import paddle
import paddle.nn

from .bytecode_transformation import transform_code_object
from .translator import InstructionTranslator


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


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


def _compile(
    frame: types.FrameType,
    compiler_fn: Callable,
) -> GuardedCode:
    f_code = frame.f_code

    def transform(instructions, code_options):
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

    logging.debug(f"\ntransformed_code:")
    [logging.debug(x) for x in list(dis.get_instructions(out_code))]

    # debug, no trace
    # return None

    g = GuardedCode(out_code)
    return g


def convert_frame(compiler_fn: callable):
    def _fn(frame: types.FrameType):
        if not has_tensor_in_frame(frame):
            return None

        return _compile(frame, compiler_fn)

    return _fn
