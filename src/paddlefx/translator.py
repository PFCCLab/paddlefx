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

__all__ = ['OutputGraph', 'Instruction', 'InstructionTranslator', 'convert_instruction']


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
        args = self.popn(2)
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    return _binary


def _unary_constructor(op_name: str):
    def _unary(self, inst: Instruction):
        op = getattr(operator, op_name)
        res = self.output.create_node('call_function', op, self.pop(), {})
        self.push(res)

    return _unary


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
    'is_': 'IS_OP',
}

UNARY_MAPPER = {'not_': 'UNARY_NOT', 'inv': 'UNARY_INVERT'}

NOT_IMPLEMENT = {
    'and_': 'BINARY_AND',
    'or_': 'BINARY_OR',
    'xor': 'BINARY_XOR',
    'iand': 'INPLACE_AND',
    'ior': 'INPLACE_OR',
    'ixor': 'INPLACE_XOR',
}


OP_MAPPER = [BINARY_MAPPER, UNARY_MAPPER, NOT_IMPLEMENT]
CONSTRUCTOR = [_binary_constructor, _unary_constructor, _not_implemented]


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
        for k, _ in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)

    def call_user_compiler(self, gl):
        compiled_fn = self.compiler_fn(gl)
        return compiled_fn

    def compile_subgraph(self):
        # add output node
        stack_values = list(self.stack)
        self.output.create_node('output', 'output', stack_values, {})
        if self.frame.f_locals.get('self', None):
            root = self.frame.f_locals.get('self')
        else:
            root = paddle.nn.Layer()
        gl = GraphLayer(root, self.output.graph)
        self.call_user_compiler(gl)

    def pop(self):
        return self.stack.pop()

    def push(self, item):
        return self.stack.append(item)

    def popn(self, n: int, reverse=True):
        assert n >= 0
        if not n:
            return []
        if reverse:
            return list(reversed([self.pop() for _ in range(n)]))
        else:
            return [self.pop() for _ in range(n)]

    def call_function(self, fn, args, kwargs):
        is_custom_call = False
        for arg in args:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break
        for arg in kwargs:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break

        # TODO: add `self.call_function` to handle more functions
        if fn is print:
            self.push(None)
        elif fn is isinstance:
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.push(res)
        elif fn.__module__.startswith("paddle"):
            if hasattr(fn, "forward"):
                fn = fn.forward
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.push(res)
        elif is_custom_call:
            raise NotImplementedError(f"custom_call is not supported")
        else:
            raise NotImplementedError(f"call function {fn} is not supported")

    def LOAD_GLOBAL(self, inst: Instruction):
        name = inst.argval
        if name in self.frame.f_globals:
            self.push(self.frame.f_globals[name])
        elif name in self.frame.f_builtins:
            self.push(self.frame.f_builtins[name])
        else:
            raise Exception(f"name '{name}' is not found")

    def POP_JUMP_IF_FALSE(self, inst: Instruction):
        pass

    def POP_JUMP_IF_TRUE(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        value = inst.argval
        self.push(value)

    def LOAD_ATTR(self, inst: Instruction):
        obj = self.pop()
        if isinstance(obj, Proxy) and obj.node.name.startswith("self"):
            res = self.output.create_node('get_param', inst.argval)
            self.push(res)
        elif hasattr(obj, inst.argval):
            value = getattr(obj, inst.argval)
            self.push(value)
        else:
            self.push(None)

    def LOAD_METHOD(self, inst: Instruction):
        target = self.pop()
        if isinstance(target, str) and target.startswith("self"):
            fn = f"{target}.{inst.argval}"
        elif isinstance(target, Proxy) and target.node.name.startswith("self"):
            fn = f"{target.node.name}.{inst.argval}"
        else:
            fn = getattr(target, inst.argval)
        self.push(fn)

    def CALL_METHOD(self, inst: Instruction):
        args = self.popn(inst.argval)
        fn = self.pop()
        if isinstance(fn, str):
            if fn.startswith("self"):
                res = self.output.create_node('call_module', fn[5:], args, {})
            else:
                # TODO(zrr1999) call_method is not implemented.
                raise NotImplemented
                # res = self.output.create_node('call_method', fn, args, {})
            self.push(res)
        else:
            if hasattr(fn, "forward"):
                fn = fn.forward
            if fn is not None:
                res = self.output.create_node('call_function', fn, args, {})
                self.push(res)
            else:
                self.push(None)

    def CALL_FUNCTION(self, inst: Instruction):
        args = self.popn(inst.argval)
        fn = self.pop()
        self.call_function(fn, args, {})

    def CALL_FUNCTION_KW(self, inst: Instruction):
        argnames = self.pop()
        args = self.popn(inst.argval)
        fn = self.pop()
        args, kwargs = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs))
        self.call_function(fn, args, kwargs)

    def BUILD_TUPLE(self, inst):
        items = self.popn(inst.argval)
        self.push(tuple(items))

    def BUILD_LIST(self, inst):
        items = self.popn(inst.argval)
        self.push(items)

    def BUILD_MAP(self, inst):
        items = self.popn(inst.argval * 2)
        result = dict()
        for k, v in zip(items[::2], items[1::2]):
            result[k] = v
        assert len(result) == len(items) / 2
        self.push(result)

    def BUILD_CONST_KEY_MAP(self, inst):
        # TODO(zrr1999): add assert
        keys = self.pop()
        values = self.popn(inst.argval)
        self.push(dict(zip(keys, values)))

    def BINARY_SUBSCR(self, inst):
        idx = self.pop()
        root = self.pop()
        res = self.output.create_node('call_method', "__getitem__", [root, idx], {})
        self.push(res)

    def STORE_SUBSCR(self, inst):
        value = self.pop()
        idx = self.pop()
        root = self.pop()
        self.output.create_node('call_method', "__setitem__", [root, idx, value], {})

    def POP_TOP(self, inst: Instruction):
        value = self.pop()

    def STORE_FAST(self, inst: Instruction):
        self.f_locals[inst.argval] = self.pop()

    def LOAD_FAST(self, inst: Instruction):
        self.push(self.f_locals[inst.argval])

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
        args = self.popn(2)
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)


for mapper, constructor in zip(OP_MAPPER, CONSTRUCTOR):
    for op_name, func_name in mapper.items():
        func = constructor(op_name)
        func = types.FunctionType(
            func.__code__, globals(), None, None, func.__closure__
        )
        setattr(InstructionTranslatorBase, func_name, func)


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
