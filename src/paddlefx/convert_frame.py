from __future__ import annotations

import dataclasses
import logging
import types

from typing import Callable

from .bytecode_transformation import Instruction, transform_code_object
from .paddle_utils import Tensor, skip_paddle_filename, skip_paddle_frame
from .pyeval import PyEval
from .utils import log_bytecode, log_code


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType


def skip_frame(frame: types.FrameType) -> bool:
    if skip_paddle_filename(frame.f_code.co_filename):
        return True

    if skip_paddle_frame(frame):
        return True

    for v in frame.f_locals.values():
        if isinstance(v, Tensor):
            return False

    return True


def convert_frame(frame: types.FrameType, compiler_fn: Callable) -> GuardedCode | None:
    if skip_frame(frame):
        logging.debug(f"skip_frame: {frame}")
        return None

    def transform(instructions: list[Instruction], code_options: dict):
        tracer = PyEval(instructions, frame, code_options, compiler_fn)
        tracer.run()

        code_options.update(tracer.output.code_options)
        instructions[:] = tracer.output.instructions

    logging.info(f"convert_frame: {frame}")
    code = frame.f_code
    log_code(code, "RAW_BYTECODE")

    # TODO: rm torch code dependency
    out_code = transform_code_object(code, transform)
    log_bytecode(
        "NEW_BYTECODE", code.co_name, code.co_filename, code.co_firstlineno, out_code
    )

    g = GuardedCode(out_code)
    return g
