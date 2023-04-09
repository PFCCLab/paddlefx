from __future__ import annotations

import dataclasses
import dis
import operator
import types

from typing import Any

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


def _binary_constructor(op_name: str):
    def _binary(self, inst: Instruction):
        op = getattr(operator, op_name)
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', op, args, {})
        self.stack.append(res)

    return _binary


def _unary_constructor(op_name: str):
    def _unary(self, inst: Instruction):
        op = getattr(operator, op_name)
        res = self.output.create_node('call_function', op, self.stack.pop(), {})
        self.stack.append(res)

    return _unary


def _f(self, inst):
    pass


def _not_implemented(op_name):
    def _not_impl(self, inst):
        raise NotImplementedError()

    return _not_impl


BINARY_MAPPER = {
    'add': 'BINARY_ADD',
    'sub': 'BINARY_SUBTRACT',
    'mul': 'BINARY_MULTIPLY',
    'floordiv': 'BINARY_FLOOR_DIVIDE',
    # NOTE: in fact, paddle doesn't support floor_divide
    'truediv': 'BINARY_TRUE_DIVIDE',
    'mod': 'BINARY_MOD',
    'pow': 'BINARY_POWER',
    'matmul': 'BINARY_MATMUL',
    'getitem': 'BINARY_GETITEM',
    'lshift': 'BINARY_LSHIFT',
    'rshift': 'BINARY_RSHIFT',
    'iadd': 'INPLACE_ADD',
    'ifloordiv': 'INPLACE_FLOOR_DIVIDE',
    'imod': 'INPLACE_MOD',
    'imul': 'INPLACE_MULTIPLY',
    'imatmul': 'INPLACE_MATRIX_MULTIPLY',
    'ipow': 'INPLACE_POWER',
    'isub': 'INPLACE_SUBTRACT',
    'itruediv': 'INPLACE_TRUE_DIVIDE',
}

UNARY_MAPPER = {'not_': 'UNARY_NOT', 'inv': 'UNARY_INVERT'}

PASS_FUNC = [
    'LOAD_GLOBAL',
    'LOAD_METHOD',
    'CALL_METHOD',
    'CALL_FUNCTION',
    'CALL_FUNCTION_KW',
    'POP_TOP',
    'MAKE_FUNCTION',
    'BINARY_SUBSCR',
    'LOAD_DEREF',
]

NOT_IMPLEMENT = {
    'and_': 'BINARY_AND',
    'or_': 'BINARY_OR',
    'xor': 'BINARY_XOR',
    'iand': 'INPLACE_AND',
    'ior': 'INPLACE_OR',
    'ixor': 'INPLACE_XOR',
}


class InstructionTranslatorMeta(type):
    def __new__(cls, *args, **kwargs):
        inst = type.__new__(cls, *args, **kwargs)
        mappers = [BINARY_MAPPER, UNARY_MAPPER, NOT_IMPLEMENT]
        constructors = [_binary_constructor, _unary_constructor, _not_implemented]
        for mapper, constructor in zip(mappers, constructors):
            for op_name, func_name in mapper.items():
                func = constructor(op_name)
                func = types.FunctionType(
                    func.__code__, globals(), None, None, func.__closure__
                )
                setattr(inst, func_name, func)
        for name in PASS_FUNC:
            setattr(inst, name, _f)
        return inst


class InstructionTranslatorBase(metaclass=InstructionTranslatorMeta):
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
        for k, _ in frame.f_locals.items():
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

    def POP_JUMP_IF_FALSE(self, inst: Instruction):
        pass

    def POP_JUMP_IF_TRUE(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        pass

    def STORE_FAST(self, inst: Instruction):
        self.f_locals[inst.argval] = self.stack.pop()

    def LOAD_FAST(self, inst: Instruction):
        self.stack.append(self.f_locals[inst.argval])

    def RETURN_VALUE(self, inst: Instruction):
        self.compile_subgraph()

    def COMPARE_OP(self, inst: Instruction):
        op_mapper = {
            '>': 'gt',
            '<': 'lt',
            '>=': 'ge',
            '<=': 'le',
            '==': 'eq',
            '!=': 'ne',
            'is': 'is_',
            'is not': 'is_not',
        }
        op = getattr(operator, op_mapper[inst.argval])
        args = list(reversed([self.stack.pop() for _ in range(2)]))
        res = self.output.create_node('call_function', op, args, {})
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
        print(inst.opname)
        if not hasattr(self, inst.opname):
            raise Exception(f"missing: {inst.opname}")
        getattr(self, inst.opname)(inst)

    def run(self):
        for i in self.instructions:
            print(i)
        for inst in self.instructions:
            self.step(inst)
