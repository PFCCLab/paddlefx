from __future__ import annotations

import dataclasses
import dis
import itertools
import operator
import types

from typing import Any

import paddle
import paddle.nn

from .graph_layer import GraphLayer
from .proxy import Proxy
from .symbolic_trace import Tracer
from .variables import (
    BuiltinVariable,
    CallableVariable,
    ConstantVariable,
    ModuleVariable,
    ObjectVariable,
    VariableBase,
)

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
        res = ObjectVariable(self.output.create_node('call_function', op, args, {}))
        self.push(res)

    return _binary


def _unary_constructor(op_name: str):
    def _unary(self, inst: Instruction):
        op = getattr(operator, op_name)
        res = ObjectVariable(
            self.output.create_node('call_function', op, self.pop(), {})
        )
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
        self.f_builtins: dict[str, Any] = {"print": print}  # TODO: add more builtins
        self.stack: list[VariableBase] = []
        self.symbolic_locals: dict[str, VariableBase] = {}
        self.symbolic_globals: dict[str, VariableBase] = {}
        # TODO: add vars in symbolic_globals
        for k, _ in frame.f_locals.items():
            self.f_locals[k] = self.output._proxy_placeholder(k)
            self.symbolic_locals[k] = ObjectVariable(self.f_locals[k])

    def call_user_compiler(self, gl):
        compiled_fn = self.compiler_fn(gl)
        return compiled_fn

    def compile_subgraph(self):
        # add output node
        stack_values = list(self.stack)
        self.output.create_node('output', 'output', stack_values, {})
        if not (root := self.frame.f_locals.get('self', None)):
            root = paddle.nn.Layer()
        gl = GraphLayer(root, self.output.graph)
        self.call_user_compiler(gl)

    def push(self, item):
        return self.stack.append(item)

    def pop(self) -> VariableBase:
        return self.stack.pop()

    def popn(self, n: int, reverse=True) -> list[VariableBase]:
        assert n >= 0
        if not n:
            return []
        if reverse:
            return list(reversed([self.pop() for _ in range(n)]))
        else:
            return [self.pop() for _ in range(n)]

    def call_function(
        self,
        fn: CallableVariable | ObjectVariable,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ):
        assert isinstance(fn, CallableVariable | ObjectVariable)
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        assert all(
            isinstance(x, VariableBase) for x in itertools.chain(args, kwargs.values())
        )
        self.push(ObjectVariable(fn.call_function(self, args, kwargs)))
        print(self.stack)

        # is_custom_call = False
        # for arg in args:
        #     if isinstance(arg, (Proxy, paddle.Tensor)):
        #         is_custom_call = True
        #         break
        # for arg in kwargs:
        #     if isinstance(arg, (Proxy, paddle.Tensor)):
        #         is_custom_call = True
        #         break

        # # TODO: add `self.call_function` to handle more functions
        # if fn is print:
        #     self.push(None)
        # elif isinstance(fn, Attribute):
        #     self.push(fn(*args, **kwargs))
        # elif fn is isinstance:
        #     res = self.output.create_node('call_function', fn, args, kwargs)
        #     self.push(res)
        # elif fn.__module__.startswith("paddle"):
        #     if hasattr(fn, "forward"):
        #         fn = fn.forward
        #     res = self.output.create_node('call_function', fn, args, kwargs)
        #     self.push(res)
        # elif is_custom_call:
        #     raise NotImplementedError(f"custom_call is not supported")
        # else:
        #     raise NotImplementedError(f"call function {fn} is not supported")

    def LOAD_GLOBAL(self, inst: Instruction):
        name = inst.argval
        if name in self.symbolic_globals:
            var = self.symbolic_globals[name]
        elif name in self.frame.f_globals:
            var = ModuleVariable(self.frame.f_globals[name])
        elif name in self.f_builtins:
            val = self.f_builtins[inst.argval]
            if callable(val):
                var = BuiltinVariable(val)
            else:
                var = ConstantVariable(val)
        else:
            raise Exception(f"name '{name}' is not found")
        self.push(var)

    def POP_JUMP_IF_FALSE(self, inst: Instruction):
        pass

    def POP_JUMP_IF_TRUE(self, inst: Instruction):
        pass

    def LOAD_CONST(self, inst: Instruction):
        value = ConstantVariable(inst.argval)
        self.push(value)

    def LOAD_ATTR(self, inst: Instruction):
        name = inst.argval
        obj = self.pop()
        if isinstance(obj, ModuleVariable):
            res = getattr(obj, name)
            self.push(res)
        elif isinstance(obj, ObjectVariable):
            # if isinstance(getattr(obj.obj, name), paddle.nn.Layer):
            res = BuiltinVariable(getattr).call_function(
                self, [obj, ConstantVariable(name)], {}
            )
            self.push(res)
        elif isinstance(obj, Proxy) and obj.node.name.startswith("self"):
            res = self.output.create_node('get_param', inst.argval)
            self.push(res)
        elif hasattr(obj, inst.argval):
            value = getattr(obj, inst.argval)
            self.push(value)
        elif hasattr(obj, inst.argval):
            value = getattr(obj, inst.argval)
            self.push(value)
        else:
            self.push(None)

    def LOAD_METHOD(self, inst: Instruction):
        self.LOAD_ATTR(inst)

    def CALL_METHOD(self, inst: Instruction):
        args = self.popn(inst.argval)
        fn = self.pop()
        assert isinstance(fn, CallableVariable | ObjectVariable)
        self.call_function(fn, args, {})
        # if isinstance(fn, Attribute):
        #     fn_name = repr(fn)
        #     if fn_name.startswith("self"):
        #         res = self.output.create_node('call_module', fn.attr, args, {})
        #     else:
        #         res = fn(*args)
        #     self.push(res)
        # else:
        #     # TODO(zrr1999) other class should be handled separately.
        #     if hasattr(fn, "forward"):
        #         fn = fn.forward
        #     if fn is not None:
        #         res = self.call_function(fn, args, {})
        #         self.push(res)
        #     else:
        #         self.push(None)

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
        keys = self.pop()
        values = self.popn(inst.argval)
        self.push(dict(zip(keys, values)))

    def BINARY_SUBSCR(self, inst):
        idx = self.pop()
        root = self.pop()
        if isinstance(root, ObjectVariable):
            res = root.call_method(self, "__getitem__", [root, idx], {})
            self.push(res)
        else:
            # TODO
            raise NotImplementedError(f"type({root}) = {type(root)}")

    def STORE_SUBSCR(self, inst):
        value = self.pop()
        idx = self.pop()
        root = self.pop()
        if isinstance(root, ObjectVariable):
            res = root.call_method(self, "__setitem__", [root, idx, value], {})
            self.push(res)
        else:
            # TODO
            raise NotImplementedError(f"type({root}) = {type(root)}")

    def POP_TOP(self, inst: Instruction):
        value = self.pop()

    def STORE_FAST(self, inst: Instruction):
        self.symbolic_locals[inst.argval] = self.pop()

    def LOAD_FAST(self, inst: Instruction):
        var = self.symbolic_locals[inst.argval]
        print(var, type(var))
        self.push(var)

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

    # note: python3.9+
    def IS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.popn(2)
        if invert:
            op = operator.is_
        else:
            op = operator.is_not
        res = self.output.create_node('call_function', op, args, {})
        self.push(res)

    def CONTAINS_OP(self, inst: Instruction):
        invert = inst.argval
        args = self.popn(2)
        if invert:
            op = operator.contains
        else:
            op = lambda a, b: b not in a
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
            print(inst)
            self.step(inst)
