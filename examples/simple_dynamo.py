import dataclasses
import dis
import operator
import types

from typing import Any, Callable, Dict, List, Optional, Tuple

import paddle
import paddle.nn

from paddlefx import GraphLayer, Tracer
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


class OutputGraph(Tracer):
    def __init__(self):
        super().__init__()


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


def _compile(frame: types.FrameType, compiler_fn: Callable):
    # TODO: add a method for frame skiping
    if frame.f_code.co_name not in ['func', 'add']:
        return None

    code = frame.f_code
    instructions = list(map(convert_instruction, dis.get_instructions(code)))

    tracer = InstructionTranslator(instructions, frame, compiler_fn)
    tracer.run()

    # TODO: not work, only support trace, but raw code cannot run(need cache support)
    g = GuardedCode(code)
    return g


def my_compiler(gm: GraphLayer, example_inputs: List[paddle.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


class DynamoContext:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        set_eval_frame(self.callback)

    def __exit__(self):
        set_eval_frame(None)

    def __call__(self, fn):
        def _fn(*args, **kwargs):
            set_eval_frame(self.callback)

            fn(*args, **kwargs)

            set_eval_frame(None)

        return _fn


def convert_frame_assert(compiler_fn: Callable):
    def _convert_frame_assert(frame: types.FrameType):
        return _compile(frame, compiler_fn)

    return _convert_frame_assert


def optimize(backend=None):
    def convert_frame(compiler_fn):
        inner_convert = convert_frame_assert(compiler_fn)

        def _convert_frame(frame: types.FrameType):
            result = inner_convert(frame)
            return result

        return _convert_frame

    return DynamoContext(convert_frame(backend))


@optimize(my_compiler)
def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@optimize(my_compiler)
def func(a=1, b=3):
    print('\tcall func')
    c = add(a, b)
    d = add(c, a)
    return d


# func(1, 3)
res = add(1, 3)

print(res)
