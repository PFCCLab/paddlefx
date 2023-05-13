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
        # self.clear_tos()

    def make_call_generated_code(self, fn_name: str) -> list[Instruction]:
        """Call the generated code function stored in fn_name."""
        # self.extend_output(self.load_function_name(fn_name, True))

        # graphargs = self.tx.output.graphargs
        # for arg in graphargs:
        #     self.extend_output(arg.load(self))

        # self.extend_output(create_call_function(len(graphargs), False))

        # LOAD_GLOBAL
        # CALL_FUNCTION
        out = []
        out.append(
            Instruction(
                opcode=dis.opmap["LOAD_GLOBAL"],
                opname="LOAD_GLOBAL",
                arg=None,
                argval=fn_name,
            )
        )
        out.append(
            Instruction(
                opcode=dis.opmap["CALL_FUNCTION"],
                opname="CALL_FUNCTION",
                arg=None,
                argval=None,
            )
        )

        self.extend_output(out)

        return out

    def get_instructions(self):
        return self._output
