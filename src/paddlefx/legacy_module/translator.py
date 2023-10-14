from __future__ import annotations

import operator
import types

from typing import Any

import paddle
import paddle.nn

from loguru import logger

from ..bytecode_transformation import Instruction, create_instruction
from ..output_graph import OutputGraph
from ..proxy import Attribute, Proxy
from ..variable_stack import VariableStack
from ..variables.base import VariableBase


def _binary_constructor(op_name: str):
    def _binary(self, inst: Instruction):
        op = getattr(operator, op_name)
        args = self.stack.pop_n(2)
        res = self.output.create_node('call_function', op, args, {})
        self.stack.push(res)

    return _binary


def _unary_constructor(op_name: str):
    def _unary(self, inst: Instruction):
        op = getattr(operator, op_name)
        res = self.output.create_node('call_function', op, self.stack.pop(), {})
        self.stack.push(res)

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
        *,
        instructions: list[Instruction],
        frame: types.FrameType,
        output: OutputGraph,
    ):
        self.instructions: list[Instruction] = instructions
        self.frame: types.FrameType = frame
        self.output: OutputGraph = output

        self.f_globals: dict[str, VariableBase] = {}
        self.f_locals: dict[str, VariableBase] = {}
        self.f_builtins: dict[str, VariableBase] = {}
        # TODO(zrr1999): self.stack should be a stack of VariableBase
        self.stack: VariableStack[VariableBase | Any] = VariableStack()
        for k, _ in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)

        for k, v in frame.f_globals.items():
            self.f_globals[k] = VariableBase(v)

    def call_function(self, fn, args, kwargs):
        # TODO: implement InlineTranslator
        is_custom_call = False
        for arg in args:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break
        for arg in kwargs:
            if isinstance(arg, (Proxy, paddle.Tensor)):
                is_custom_call = True
                break

        if isinstance(fn, Attribute):
            self.stack.push(fn(*args, **kwargs))
        elif fn is isinstance:
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.stack.push(res)
        elif fn.__module__.startswith("paddle"):
            if hasattr(fn, "forward"):
                fn = fn.forward
            res = self.output.create_node('call_function', fn, args, kwargs)
            self.stack.push(res)
        elif is_custom_call:
            raise NotImplementedError(f"custom_call is not supported")
        else:
            raise NotImplementedError(f"call function {fn} is not supported")

    def LOAD_GLOBAL(self, inst: Instruction):
        name = inst.argval
        if name in self.f_globals:
            self.stack.push(self.f_globals[name].value)
        elif name in self.f_builtins:
            self.stack.push(self.f_builtins[name].value)
        else:
            raise Exception(f"name '{name}' is not found")

    def POP_JUMP_IF_FALSE(self, inst: Instruction):
        pass

    def POP_JUMP_IF_TRUE(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        value = inst.argval
        self.stack.push(value)

    def LOAD_ATTR(self, inst: Instruction):
        obj = self.stack.pop()
        if isinstance(obj, Proxy) and obj.node.name.startswith("self"):
            res = self.output.create_node('get_param', inst.argval)
            self.stack.push(res)
        elif hasattr(obj, inst.argval):
            value = getattr(obj, inst.argval)
            self.stack.push(value)
        else:
            self.stack.push(None)

    def LOAD_METHOD(self, inst: Instruction):
        target = self.stack.pop()
        fn = getattr(target, inst.argval)
        self.stack.push(fn)

    def CALL_METHOD(self, inst: Instruction):
        args = self.stack.pop_n(inst.argval)
        fn = self.stack.pop()
        if isinstance(fn, Attribute):
            fn_name = repr(fn)
            if fn_name.startswith("self"):
                res = self.output.create_node('call_module', fn.attr, args, {})
            else:
                res = fn(*args)
            self.stack.push(res)
        else:
            # TODO(zrr1999) other class should be handled separately.
            if hasattr(fn, "forward"):
                fn = fn.forward
            if fn is not None:
                res = self.call_function(fn, args, {})
                self.stack.push(res)
            else:
                self.stack.push(None)

    def CALL_FUNCTION(self, inst: Instruction):
        args = self.stack.pop_n(inst.argval)
        fn = self.stack.pop()
        self.call_function(fn, args, {})

    def CALL_FUNCTION_KW(self, inst: Instruction):
        argnames = self.stack.pop()
        args = self.stack.pop_n(inst.argval)
        fn = self.stack.pop()
        args, kwargs = args[: -len(argnames)], args[-len(argnames) :]
        kwargs = dict(zip(argnames, kwargs))
        self.call_function(fn, args, kwargs)

    def BUILD_TUPLE(self, inst):
        items = self.stack.pop_n(inst.argval)
        self.stack.push(tuple(items))

    def BUILD_LIST(self, inst):
        items = self.stack.pop_n(inst.argval)
        self.stack.push(items)

    def BUILD_MAP(self, inst):
        items = self.stack.pop_n(inst.argval * 2)
        result = dict()
        for k, v in zip(items[::2], items[1::2]):
            result[k] = v
        assert len(result) == len(items) / 2
        self.stack.push(result)

    def BUILD_CONST_KEY_MAP(self, inst):
        keys = self.stack.pop()
        values = self.stack.pop_n(inst.argval)
        self.stack.push(dict(zip(keys, values)))

    def BINARY_SUBSCR(self, inst):
        idx = self.stack.pop()
        root = self.stack.pop()
        if isinstance(root, Proxy):
            res = root[idx]
        else:
            res = self.output.create_node('call_method', "__getitem__", [root, idx], {})
        self.stack.push(res)

    def STORE_SUBSCR(self, inst):
        value = self.stack.pop()
        idx = self.stack.pop()
        root = self.stack.pop()
        self.output.create_node('call_method', "__setitem__", [root, idx, value], {})

    def POP_TOP(self, inst: Instruction):
        value = self.stack.pop()

    def STORE_FAST(self, inst: Instruction):
        self.f_locals[inst.argval] = self.stack.pop()

    def LOAD_FAST(self, inst: Instruction):
        self.stack.push(self.f_locals[inst.argval])

    def RETURN_VALUE(self, inst: Instruction):
        self.output.compile_subgraph(self)
        self.output.add_output_instructions(
            [
                create_instruction("RETURN_VALUE"),
            ]
        )

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
        args = self.stack.pop_n(2)
        res = self.output.create_node('call_function', op, args, {})
        self.stack.push(res)

    # note: python3.9+
    def IS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.stack.pop_n(2)
        if invert:
            op = operator.is_
        else:
            op = operator.is_not
        res = self.output.create_node('call_function', op, args, {})
        self.stack.push(res)

    def CONTAINS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.stack.pop_n(2)
        if invert:
            op = operator.contains
        else:
            op = lambda a, b: b not in a
        res = self.output.create_node('call_function', op, args, {})
        self.stack.push(res)


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
        *,
        instructions: list[Instruction],
        frame: types.FrameType,
        code_options: dict,
        compiler_fn: Any,
    ):
        output = OutputGraph(
            f_globals=frame.f_globals,
            code_options=code_options,
            compiler_fn=compiler_fn,
        )
        super().__init__(
            instructions=instructions,
            frame=frame,
            output=output,
        )

    def step(self, inst: Instruction):
        if not hasattr(self, inst.opname):
            raise NotImplementedError(f"missing: {inst.opname}")

        logger.debug(f"TRACE {inst.opname} {inst.argval} {self.stack}")
        getattr(self, inst.opname)(inst)

    def run(self):
        # TODO: support graph break
        for inst in self.instructions:
            self.step(inst)
