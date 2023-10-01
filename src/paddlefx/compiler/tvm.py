from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle
import paddle.device

import paddlefx

from .base import CompilerBase, CompilerError, paddle_dtype_to_str

if TYPE_CHECKING:
    from tvm import te


class TVMCompiler(CompilerBase):
    def gen_compiled_func(
        self, symbol_table: dict[str, te.Tensor], dummy_inputs: list, dummy_outputs: Any
    ):
        import tvm

        from tvm import te

        device = paddle.device.get_device()
        if device == "cpu":
            target = tvm.target.Target(target="llvm", host="llvm")
        elif device == "gpu":
            target = tvm.target.Target(target="cuda", host="llvm")
        else:
            raise ValueError(f"Unsupported device in tvm backend: {device}")
        schedule = te.create_schedule(symbol_table["output"].op)
        tvm_func = tvm.build(
            schedule,
            [
                v
                for k, v in symbol_table.items()
                if v.name.startswith("input") or k == "output"
            ],
            target,
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

        compiled_func(*dummy_inputs)
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

    def compile_call_module(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        pass

        target_name = node.target
        raise CompilerError(f"Unsupported module: {target_name}")

    def compile_call_function(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        from tvm import topi

        target_name = node.target.__name__

        map_ops_to_tvm = {
            "add": topi.add,
            "sub": topi.subtract,
            "subtract": topi.subtract,
            "mul": topi.multiply,
            "truediv": topi.divide,
            "gt": topi.greater,
            "lt": topi.less,
            "ge": topi.greater_equal,
            "le": topi.less_equal,
        }

        if target_name in map_ops_to_tvm.keys():
            symbol_args = [symbol_table[str(arg)] for arg in node.args]
            symbol_table[node.name] = map_ops_to_tvm[target_name](*symbol_args)
        else:
            raise NotImplementedError(f"Unsupported function: {target_name}")

    def compile_output(
        self, node: paddlefx.Node, symbol_table: dict[str, te.Tensor], inputs: list
    ):
        ret = symbol_table[str(node.args[0][0])]
        symbol_table["output"] = ret
