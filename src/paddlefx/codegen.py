from __future__ import annotations

import dis

from typing import TYPE_CHECKING

from .translator import Instruction

if TYPE_CHECKING:
    from .translator import Instruction, InstructionTranslatorBase


class PyCodegen:
    def __init__(
        self,
        tx: InstructionTranslatorBase = None,
    ):
        self.tx = tx

        self._output: list[Instruction] = []

    def extend_output(self, insts):
        assert all(isinstance(x, Instruction) for x in insts)
        self._output.extend(insts)

    def make_call_generated_code(self, fn_name: str):
        load_function = Instruction(
            opcode=dis.opmap["LOAD_GLOBAL"],
            opname="LOAD_GLOBAL",
            arg=False,
            argval=fn_name,
        )
        self.extend_output([load_function])

        placeholders = self.tx.output.placeholders
        for x in placeholders:
            load_fast = Instruction(
                opcode=dis.opmap["LOAD_FAST"],
                opname="LOAD_FAST",
                arg=None,
                argval=x.name,
            )
            self.extend_output([load_fast])

        call_function = Instruction(
            opcode=dis.opmap["CALL_FUNCTION"],
            opname="CALL_FUNCTION",
            arg=len(placeholders),
            argval=None,
        )
        self.extend_output([call_function])

    def get_instructions(self):
        return self._output
