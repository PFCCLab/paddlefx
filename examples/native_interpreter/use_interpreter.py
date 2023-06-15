from __future__ import annotations

import paddle

from build import myinterpreter

import paddlefx

from paddlefx import symbolic_trace


def net(a, b):
    c = paddle.add(a, b)
    d = paddle.multiply(c, a)
    e = paddle.add(c, d)
    return e


traced_layer = symbolic_trace(net)

print(f">> python IR for {net.__name__}")
traced_layer.graph.print_tabular()

# the very simple IR we want to lower fx graph to
# each instruction is a list of string of:  operation, left_operand, right_operand, result
# only two op supported: add, mul
input_names = []
instructions = []

target_to_name = {paddle.add: "add", paddle.multiply: "mul"}


def lower_to_native_interpreter(orig_net):
    # step1: trace the net
    graph_layer = symbolic_trace(orig_net)

    # step2: lower fx IR to native interpreter's instructions format
    for n in graph_layer.graph.nodes:
        target, args, out_name = n.target, n.args, n.name
        assert len(n.kwargs) == 0, "kwargs currently not supported"

        if n.op == 'placeholder':
            input_names.append(target)
        elif n.op == 'call_function':
            assert target in target_to_name, "Unsupported call target " + target
            arg_names = []
            for arg in args:
                if not isinstance(arg, paddlefx.Node):
                    raise RuntimeError('Unsupported arg' + arg)
                else:
                    arg_names.append(arg.name)
            instructions.append(
                [target_to_name[target], arg_names[0], arg_names[1], out_name]
            )
        elif n.op == 'output':
            # not handled output node for now
            pass
        else:
            raise RuntimeError('Unsupported opcode ' + n.op)


lower_to_native_interpreter(net)
print(f">> lowered native interpreter IR for {net.__name__}")
print(input_names)

list(map(print, instructions))

input_a = 3
input_b = 4

print(f">> paddle running result of {net.__name__}")
# not figured out howto pass in paddle::Tensor to paddle's C++ API
a = paddle.full([3, 4], input_a)
b = paddle.full([3, 4], input_b)
print(net(a, b))

print(f">> native interpreter running process of {net.__name__}")
myinterpreter.set_input_names(input_names)
myinterpreter.set_instructions(instructions)
myinterpreter.execute(input_a, input_b)
