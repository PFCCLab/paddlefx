from __future__ import annotations

import dis
import os
import sys
import traceback
import types

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .bytecode_transformation import Instruction

logger.remove()
logger.add(
    sys.stdout, level=os.environ.get("LOG_LEVEL", "INFO")
)


def format_bytecode(prefix, name, filename, line_no, code):
    return f"{prefix} {name} {filename} line {line_no} \n{dis.Bytecode(code).dis()}"


def log_bytecode(prefix, name, filename, line_no, code, log_fn=logger.info):
    log_fn(format_bytecode(prefix, name, filename, line_no, code))


def log_code(code: types.CodeType, prefix='', log_fn=logger.info):
    log_bytecode(
        prefix, code.co_name, code.co_filename, code.co_firstlineno, code, log_fn=log_fn
    )


def format_instruction(inst: dis.Instruction | Instruction):
    if inst.offset is None:
        if inst.arg is None:
            return f"{'': <15} {inst.opname: <25} {'': <2} ({inst.argval})"
        else:
            return f"{'': <15} {inst.opname: <25} {inst.arg: <2} ({inst.argval})"
    else:
        return f"{'' : <12} {inst.offset} {inst.opname} {inst.argval}"


def log_instructions(
    instructions: list[dis.Instruction] | list[Instruction],
    prefix='',
    log_fn=logger.info,
):
    log_fn(f"{prefix}")
    for inst in instructions:
        log_fn(format_instruction(inst))


def get_instructions(code: types.CodeType):
    return list(dis.get_instructions(code))


def hashable(obj) -> bool:
    try:
        hash(obj)
        return True
    except TypeError as e:
        return False


class InnerErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: add BreakpointManager

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(InnerErrorBase):
    pass


class HasNoAttributeError(InnerError):
    pass


class FallbackError(InnerErrorBase):
    def __init__(self, msg, disable_eval_frame=False):
        super().__init__(msg)
        self.disable_eval_frame = False


# raise in inline function call strategy.
class BreakGraphError(InnerErrorBase):
    pass
