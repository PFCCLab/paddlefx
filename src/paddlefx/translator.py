from __future__ import annotations

import dataclasses
import dis
import operator
import types

from typing import Any

import paddle
import paddle.nn

from .graph_layer import GraphLayer
from .proxy import Proxy
from .symbolic_trace import Tracer


class OutputGraph(Tracer):
    def __init__(self):
        super().__init__()


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction."""

    opcode: int
    opname: str
    arg: int | None
    argval: Any
    offset: int | None = None
    starts_line: int | None = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Instruction | None = None

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
        instructions: list[Instruction],
        frame: types.FrameType,
        compiler_fn: Any,
        output: OutputGraph,
    ):
        self.instructions: list[Instruction] = instructions
        self.frame: types.FrameType = frame
        self.compiler_fn = compiler_fn
        self.output: OutputGraph = output

        self.f_locals = {}
        self.stack = []
        for k, v in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)

    def call_user_compiler(self, gl):
        compiled_fn = self.compiler_fn(gl)
        return compiled_fn

    def compile_subgraph(self):
        # add output node
        stack_values = list(self.stack)
        self.output.create_node('output', 'output', stack_values, {})

        gl = GraphLayer(paddle.nn.Layer(), self.output.graph)
        self.call_user_compiler(gl)

    def LOAD_GLOBAL(self, inst: Instruction):
        name = inst.argval
        if name in self.frame.f_globals:
            self.stack.append(self.frame.f_globals[name])
        elif name in self.frame.f_builtins:
            self.stack.append(self.frame.f_builtins[name])
        else:
            raise Exception(f"name '{name}' is not found")

    def LOAD_CONST(self, inst: Instruction):
        value = inst.argval
        self.stack.append(value)

    def CALL_FUNCTION(self, inst: Instruction):
        args = [self.stack.pop() for _ in range(inst.argval)]
        fn = self.stack.pop()

        is_custom_call = False
        for arg in args:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break

        # TODO: add `self.call_function` to handle more functions
        if fn == print:
            self.stack.append(None)
        elif is_custom_call:
            raise NotImplementedError(f"custom_call is not supported")
        else:
            raise NotImplementedError(f"call function {fn} is not supported")

    def POP_TOP(self, inst: Instruction):
        value = self.stack.pop()

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

    def BINARY_SUBTRACT(self, inst: Instruction):
        sub = getattr(operator, 'sub')
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', sub, args, {})
        self.stack.append(res)

    def BINARY_MULTIPLY(self, inst: Instruction):
        mul = getattr(operator, 'mul')
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', mul, args, {})
        self.stack.append(res)

    # NOTE: in fact, paddle doesn't support floor_divide
    def BINARY_FLOOR_DIVIDE(self, inst: Instruction):
        floordiv = getattr(operator, 'floordiv')
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', floordiv, args, {})
        self.stack.append(res)

    def BINARY_TRUE_DIVIDE(self, inst: Instruction):
        truediv = getattr(operator, 'truediv')
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', truediv, args, {})
        self.stack.append(res)


class InstructionTranslator(InstructionTranslatorBase):
    def __init__(
        self,
        instructions: list[Instruction],
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
