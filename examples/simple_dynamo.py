import dataclasses
import dis
import operator
import types

from typing import Any, Dict, List, Optional, Tuple

import opcode  # noqa

import paddlefx

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


class OutputGraph:
    def __init__(self):
        super().__init__()
        self.graph = paddlefx.Graph()


class InstructionTranslatorBase:
    def __init__(
        self,
        instructions: List[Instruction],
        frame: types.FrameType,
        output: OutputGraph,
    ):
        self.instructions: List[Instruction] = instructions
        self.frame: types.FrameType = frame
        self.output: OutputGraph = output

        self.f_locals = {}
        self.stack = []
        for k, v in frame.f_locals.items():
            node = self.output.graph.placeholder(k)
            self.f_locals[k] = node

    def compile_subgraph(self):
        # add output node
        stack_values = list(self.stack)
        self.output.graph.create_node('output', 'output', stack_values)

        self.output.graph.print_tabular()

    def LOAD_GLOBAL(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        pass

    def CALL_FUNCTION(self, inst: Instruction):
        pass

    def POP_TOP(self, inst: Instruction):
        pass

    def STORE_FAST(self, inst: Instruction):
        self.f_locals[inst.argval] = self.stack.pop()

    def LOAD_FAST(self, inst: Instruction):
        self.stack.append(self.f_locals[inst.argval])

    def RETURN_VALUE(self, inst: Instruction):
        self.compile_subgraph()

    def BINARY_ADD(self, inst: Instruction):
        add = getattr(operator, 'add')
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.graph.create_node('call_function', add, 'add', args)
        self.stack.append(res)


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: List[Instruction],
        frame: types.FrameType,
    ):
        super().__init__(instructions, frame, OutputGraph())

    def step(self, inst: Instruction):
        if not hasattr(self, inst.opname):
            raise Exception(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)

    def run(self):
        for inst in self.instructions:
            self.step(inst)


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
    print(frame)
    print(dis.disassemble(frame.f_code))
    convert_frame(frame)

    f_code = frame.f_code
    g = GuardedCode(f_code)
    return g


def add(a, b):
    # print('\tcall add')
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
