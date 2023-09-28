from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import paddle

import paddlefx

if TYPE_CHECKING:
    from tvm import te


def paddle_dtype_to_str(dtype: paddle.dtype) -> str:
    if dtype == paddle.float32:
        return "float32"
    elif dtype == paddle.float64:
        return "float64"
    elif dtype == paddle.float16:
        return "float16"
    elif dtype == paddle.int32:
        return "int32"
    elif dtype == paddle.int64:
        return "int64"
    elif dtype == paddle.bool:
        return "bool"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class CompilerBase:
    def __init__(self, *, print_tabular: bool = True):
        self.print_tabular = print_tabular
        self.input_index = 0

    def __call__(self, gl: paddlefx.GraphLayer, dummy_inputs: list):
        if self.print_tabular:
            gl.graph.print_tabular()
        return self.compile(gl, dummy_inputs)

    def compile(self, gl: paddlefx.GraphLayer, dummy_inputs: list) -> Callable:
        dummy_outputs = gl.forward(*dummy_inputs)
        symbol_table: dict[str, Any] = {}
        try:
            for node in gl.graph.nodes:
                getattr(self, f"compile_{node.op}")(node, symbol_table, dummy_inputs)
            self.input_index = 0
            return self.gen_compiled_func(symbol_table, dummy_outputs)
        except (AttributeError, NotImplementedError) as e:
            print(f"AttributeError when compiling graph: {e}")
            self.input_index = 0
            return gl.forward

    def gen_compiled_func(self, symbol_table: dict[str, Any], dummy_outputs: Any):
        raise NotImplementedError("CompilerBase is a abstract class")


class TVMCompiler(CompilerBase):
    def gen_compiled_func(self, symbol_table: dict[str, te.Tensor], dummy_outputs: Any):
        import tvm

        from tvm import te

        tgt = tvm.target.Target(target="llvm", host="llvm")
        schedule = te.create_schedule(symbol_table["output"].op)
        tvm_func = tvm.build(
            schedule,
            [
                v
                for k, v in symbol_table.items()
                if v.name.startswith("input") or k == "output"
            ],
            tgt,
            name=symbol_table["output"].name,
        )

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy()) for arg in args]
            dummy_output = dummy_outputs[0]
            output = tvm.nd.empty(
                dummy_output.shape, paddle_dtype_to_str(dummy_output.dtype)
            )
            tvm_func(*inputs, output)
            output = paddle.to_tensor(output.asnumpy())
            return (output,)

        return compiled_func

    def compile_placeholder(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        from tvm import te

        symbol_table[node.name] = te.placeholder(
            inputs[self.input_index].shape,
            paddle_dtype_to_str(inputs[self.input_index].dtype),
            name=f"input_{node.name}",
        )
        self.input_index += 1

    def compile_call_function(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        from tvm import te

        if node.target.__name__ in ["add", "sub", "mul", "div"]:
            left = symbol_table[str(node.args[0])]
            right = symbol_table[str(node.args[1])]
            symbol_table[str(node.name)] = te.compute(  # type: ignore
                left.shape,
                lambda *i: node.target(left[i], right[i]),
                name=str(node.name),
            )
        else:
            raise NotImplementedError(f"Unsupported function: {node.target.__name__}")

    def compile_output(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        ret = symbol_table.get(str(node.args[0][0]))
        symbol_table["output"] = ret
