from typing import List

import paddle
import paddle.nn

import paddlefx


def my_compiler(gl: paddlefx.GraphLayer, example_inputs: List[paddle.Tensor] = None):
    print("my_compiler() called with FX graph:")
    gl.graph.print_tabular()
    return gl.forward


def add(a, b):
    print('\tcall add')
    c = a + b
    return c


@paddlefx.optimize(my_compiler, supported_ops=['add', 'func'])
def func(a, b):
    print('\tcall func')
    c = add(a, b)
    d = add(c, c)
    return d


res = func(1, 3)
print(res)
assert res == 8
