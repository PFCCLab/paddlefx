from __future__ import annotations

import collections

from functools import lru_cache
from typing import TYPE_CHECKING

from .bytecode_transformation import *  # noqa
from .source import LocalSource
from .variable_stack import VariableStack
from .variables.base import TensorVariable

if TYPE_CHECKING:
    from .graph import Node
    from .pyeval import PyEvalBase
    from .variables.base import VariableBase


@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn


class PyCodegen:
    def __init__(
        self,
        tx: PyEvalBase,
        graph_output_var: str | None = None,
    ):
        self.tx = tx
        self.code_options = self.tx.output.code_options

        self.graph_output_var = graph_output_var

        self.top_of_stack = None
        self.graph_outputs = collections.OrderedDict()
        self.instructions: list[Instruction] = []

    def clear_tos(self):
        self.top_of_stack = None

    def append_output(self, inst):
        assert isinstance(inst, Instruction)
        self.instructions.append(inst)
        self.clear_tos()

    def extend_output(self, insts):
        assert all(isinstance(x, Instruction) for x in insts)
        self.instructions.extend(insts)
        self.clear_tos()

    def create_load(self, name):
        # assert name in self.code_options["co_varnames"], f"{name} missing"
        return create_instruction("LOAD_FAST", argval=name)

    def create_store(self, name):
        # assert name in self.code_options["co_varnames"]
        return create_instruction("STORE_FAST", argval=name)

    def create_load_global(self, name, push_null):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        # assert name in self.code_options["co_names"], f"{name} not in co_names"
        return create_load_global(name, push_null)

    def create_load_const(self, value):
        return create_instruction("LOAD_CONST", argval=value)

    def create_load_attr(self, name):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        return create_instruction("LOAD_ATTR", argval=name)

    def create_load_attrs(self, names):
        return [self.create_load_attr(name) for name in names.split(".")]

    def load_function_name(self, fn_name, push_null, num_on_stack=0):
        """Load the global fn_name on the stack num_on_stack down."""
        output = []
        self.code_options["co_names"] += (fn_name,)
        output.extend(
            [
                self.create_load_global(fn_name, False),
                *self.rot_n(num_on_stack + 1),
            ]
        )
        return output

    def rot_n(self, n):
        try:
            return create_rot_n(n)
        except AttributeError:
            # desired rotate bytecode doesn't exist, generate equivalent bytecode
            return [
                create_instruction("BUILD_TUPLE", arg=n),
                self.create_load_const(rot_n_helper(n)),
                *create_rot_n(2),
                create_instruction("CALL_FUNCTION_EX", arg=0),
                create_instruction("UNPACK_SEQUENCE", arg=n),
            ]

    def create_call_function_kw(self, nargs, kw_names, push_null):
        return [
            self.create_load_const(kw_names),
            create_instruction("CALL_FUNCTION_KW", arg=nargs),
        ]

    def make_call_generated_code(self, fn_name: str):
        self.append_output(create_load_global(fn_name, False))

        placeholders: list[Node] = []
        for node in self.tx.output.graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)

        for node in placeholders:
            self.append_output(self.create_load(node.target))
        self.extend_output(create_call_function(len(placeholders), False))

    def call(self, vars: VariableStack[VariableBase]):
        for var in vars:
            self.call_one(var)

    def call_one(self, value: VariableBase):
        """Generate code such that top-of-stack (TOS) is set to value."""
        output = self.instructions
        graph_outputs = self.graph_outputs

        if self.top_of_stack is value:
            output.append(create_dup_top())
            return

        # TODO: better org
        if value.source is not None:
            if isinstance(value.source, LocalSource):
                output.append(self.create_load(value.source.local_name))
            else:
                raise Exception(f"unsupported source: {value.source}")
        elif isinstance(value, TensorVariable):
            # TODO: clean it
            graph_outputs_key = id(value)
            if graph_outputs_key not in graph_outputs:
                graph_outputs[graph_outputs_key] = value

            output.append(self.create_load(self.graph_output_var))
            # TODO: rm hardcode
            output.append(self.create_load_const(0))
            output.append(create_instruction("BINARY_SUBSCR"))
        elif value.var == None:
            output.append(self.create_load_const(None))
        elif type(value.var) == types.FunctionType:
            output.append(self.create_load_global(value.var.__name__, False))
        elif type(value.var) == types.BuiltinFunctionType:
            if value.var == print:
                output.append(self.create_load_global(value.var.__name__, False))
        elif type(value.var) in [str, bool, int]:
            output.append(self.create_load_const(value.var))
        elif type(value.var) == tuple:
            self.call(VariableStack(list(value.var)))
            output.append(create_instruction("BUILD_TUPLE", arg=len(value.var)))
        else:
            raise ValueError(f"unsupported type: {type(value.var)}")

        self.top_of_stack = value
