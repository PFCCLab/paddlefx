import dataclasses
import dis
import types

from typing import Any, Dict, List, Optional, Tuple

import opcode  # noqa

from paddlefx._eval_frame import set_eval_frame


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    # check_fn: GuardFn


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction."""

    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


class InstructionTranslatorBase:
    def __init__(
        self,
        instructions: List[Instruction],
    ):
        self.instructions: List[Instruction] = instructions

    def LOAD_GLOBAL(self, inst):
        pass

    def LOAD_CONST(self, inst):
        pass

    def CALL_FUNCTION(self, inst):
        pass

    def POP_TOP(self, inst):
        pass

    def STORE_FAST(self, inst):
        pass

    def LOAD_FAST(self, inst):
        pass

    def RETURN_VALUE(self, inst):
        pass

    def BINARY_ADD(self, inst):
        pass


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: List[Instruction],
        frame,
    ):
        super().__init__(instructions)
        self.frame: types.FrameType = frame

    def step(self, inst: Instruction):
        if not hasattr(self, inst.opname):
            raise Exception(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)

    def run(self):
        for inst in self.instructions:
            self.step(inst)
        # add output

        print()


def convert_frame(frame: types.FrameType):
    code = frame.f_code
    instructions = list(map(convert_instruction, dis.get_instructions(code)))

    tracer = InstructionTranslator(instructions, frame)
    tracer.run()


def callback(frame: types.FrameType):
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in ['func', 'add']:
        return None

    print('enter callback')
    # print(frame)
    print(dis.disassemble(frame.f_code))
    convert_frame(frame)

    f_code = frame.f_code
    g = GuardedCode(f_code)
    return g


def add(a, b):
    print('\tcall add')
    c = a + b
    return c


def func(a=1, b=3):
    print('\tcall func')
    c = add(a, b)
    d = add(c, a)
    return d


print('set_eval_frame(callback)')
set_eval_frame(callback)

# func(1, 3)
add(1, 3)

print('\nset_eval_frame(None)')
set_eval_frame(None)

# func(1, 4)
