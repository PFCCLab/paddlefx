from __future__ import annotations

import dataclasses
import logging
import types

from typing import Callable

import paddle
import paddle.nn

from .bytecode_transformation import transform_code_object
from .translator import InstructionTranslator
from .utils import format_bytecode


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
    def transform(instructions, code_options):
        tracer = InstructionTranslator(
            instructions=instructions,
            frame=frame,
            code_options=code_options,
            compiler_fn=compiler_fn,
        )
        tracer.run()

        instructions[:] = tracer.output.output_instructions

    code = frame.f_code
    out_code = transform_code_object(code, transform)

    logging.debug(
        f"{format_bytecode('raw_code', code.co_name, code.co_filename, code.co_firstlineno, code)}"
    )
    logging.debug(
        f"{format_bytecode('transformed_code', code.co_name, code.co_filename, code.co_firstlineno, out_code)}"
    )

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
