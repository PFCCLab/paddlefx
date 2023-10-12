from __future__ import annotations

import logging
import types

from typing import TYPE_CHECKING, Callable

from .bytecode_transformation import Instruction, transform_code_object
from .cache_manager import CodeCacheManager, GuardedCode
from .paddle_utils import Tensor, skip_paddle_filename, skip_paddle_frame
from .pyeval import PyEval
from .utils import log_bytecode, log_code

if TYPE_CHECKING:
    pass


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
    # TODO: guard_fn is not declared in this scope
    guard_fn = None

    def transform(instructions: list[Instruction], code_options: dict):
        tracer = PyEval(instructions, frame, code_options, compiler_fn)
        tracer.run()
        nonlocal guard_fn
        guard_fn = tracer.output.guard_fn

        code_options.update(tracer.output.code_options)
        instructions[:] = tracer.output.instructions

    logging.info(f"convert_frame: {frame}")
    code = frame.f_code
    log_code(code, "ORIGINAL_BYTECODE")

    if (cached_code := CodeCacheManager.get_cache(frame)) is not None:
        logging.info(f"cached_code: {cached_code}")
        return cached_code

    # TODO: rm torch code dependency
    out_code = transform_code_object(code, transform)
    log_bytecode(
        "NEW_BYTECODE", code.co_name, code.co_filename, code.co_firstlineno, out_code
    )
    new_code = GuardedCode(out_code, guard_fn)
    CodeCacheManager.add_cache(code, new_code)
    return new_code
