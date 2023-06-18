from __future__ import annotations

import dataclasses
import logging
import types

from typing import Callable

import paddle
import paddle.nn

from .bytecode_transformation import Instruction, transform_code_object
from .translator import InstructionTranslator
from .utils import format_bytecode


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


def skip_frame(frame: types.FrameType) -> bool:
    # skip paddle internal code
    if frame.f_code.co_filename.endswith('paddle/fluid/dygraph/math_op_patch.py'):
        return True
    elif frame.f_code.co_filename.endswith('paddle/fluid/framework.py'):
        return True
    elif frame.f_code.co_filename.endswith('paddle/tensor/to_string.py'):
        return True
    elif frame.f_code.co_filename.endswith('fluid/dygraph/varbase_patch_methods.py'):
        return True
    elif frame.f_code.co_name == 'in_dygraph_mode':
        return True

    for v in frame.f_locals.values():
        # TODO: supprt containers & more
        if isinstance(v, paddle.Tensor):
            return False

    return True


def _compile(
    frame: types.FrameType,
    compiler_fn: Callable,
) -> GuardedCode:
    def transform(instructions: list[Instruction], code_options: dict):
        tracer = InstructionTranslator(
            instructions=instructions,
            frame=frame,
            code_options=code_options,
            compiler_fn=compiler_fn,
        )
        tracer.run()

        instructions[:] = tracer.output.output_instructions
        code_options.update(tracer.output.code_options)

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
        if skip_frame(frame):
            return None

        return _compile(frame, compiler_fn)

    return _fn
