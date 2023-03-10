import dataclasses
import dis
import operator
import types

from typing import Any, List, Optional

import paddle
import paddle.nn

from .graph_layer import GraphLayer
from .symbolic_trace import Tracer


class OutputGraph(Tracer):
    def __init__(self):
        super().__init__()


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
        frame: types.FrameType,
        compiler_fn: Any,
        output: OutputGraph,
    ):
        self.instructions: List[Instruction] = instructions
        self.frame: types.FrameType = frame
        self.compiler_fn = compiler_fn
        self.output: OutputGraph = output

        self.f_locals = {}
        self.stack = []
        for k, v in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)

    def call_user_compiler(self, gm):
        compiled_fn = self.compiler_fn(gm, None)
        return compiled_fn

    def compile_subgraph(self):
        # add output node
        stack_values = list(self.stack)
        self.output.create_node('output', 'output', stack_values, {})

        gm = GraphLayer(paddle.nn.Layer(), self.output.graph)
        self.call_user_compiler(gm)

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
        res = self.output.create_node('call_function', add, args, {})
        self.stack.append(res)


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: List[Instruction],
        frame: types.FrameType,
        compiler_fn: Any,
    ):
        super().__init__(instructions, frame, compiler_fn, OutputGraph())

    def step(self, inst: Instruction):
        if not hasattr(self, inst.opname):
            raise Exception(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)

    def run(self):
        for inst in self.instructions:
            self.step(inst)
