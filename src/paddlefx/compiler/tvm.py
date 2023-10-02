from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.device

import paddlefx

from .base import CompilerBase, CompilerError, paddle_dtype_to_str

if TYPE_CHECKING:
    from tvm import te

    from .base import SymbolTable


class TVMCompiler(CompilerBase):
    def gen_compiled_func(
        self,
        gl: paddlefx.GraphLayer,
        symbol_table: SymbolTable[te.Tensor],
        dummy_inputs: list,
    ):
        import tvm

        from tvm import te

        device = paddle.device.get_device()
        if device == "cpu":
            target = tvm.target.Target(target="llvm", host="llvm")
        elif device == "gpu":
            target = tvm.target.Target(target="cuda", host="llvm")
        else:
            raise CompilerError(f"Unsupported device in tvm backend: {device}")
        schedule = te.create_schedule([out.op for out in symbol_table.outputs])
        tvm_func = tvm.build(
            schedule,
            [*symbol_table.inputs, *symbol_table.outputs],
            target,
            name="tvm_func",
        )

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy()) for arg in args]
            outputs = [
                tvm.nd.empty(out.shape, out.dtype) for out in symbol_table.outputs
            ]
            tvm_func(*inputs, *outputs)
            return tuple(paddle.to_tensor(out.asnumpy()) for out in outputs)

        return compiled_func

    def compile_placeholder(
        self, node: paddlefx.Node, symbol_table: SymbolTable[te.Tensor], inputs: list
    ):
        from tvm import te

        symbol_table.add_input(
            node.name,
            te.placeholder(
                inputs[self.input_index].shape,
                paddle_dtype_to_str(inputs[self.input_index].dtype),
                name=f"input_{node.name}",
            ),
        )
        self.input_index += 1

    def compile_call_module(
        self, node: paddlefx.Node, symbol_table: SymbolTable[te.Tensor], inputs: list
    ):
        target_name = node.target
        raise CompilerError(f"Unsupported module: {target_name}")

    def compile_call_function(
        self, node: paddlefx.Node, symbol_table: SymbolTable[te.Tensor], inputs: list
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
        self, node: paddlefx.Node, symbol_table: SymbolTable[te.Tensor], inputs: list
    ):
        symbol_table.outputs = tuple(symbol_table[str(arg)] for arg in node.args[0])
