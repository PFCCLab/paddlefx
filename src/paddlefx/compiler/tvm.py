from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.device

from paddle import nn

import paddlefx

from .base import CompilerBase, CompilerError, paddle_dtype_to_str

if TYPE_CHECKING:
    from tvm import te

    from .base import SymbolTable


class TVMCompiler(CompilerBase):
    def gen_compiled_func(self, symbol_table: SymbolTable[te.Tensor, paddle.Tensor]):
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

        weights = [tvm.nd.array(p.numpy()) for p in symbol_table.weights]

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy()) for arg in args]
            outputs = [
                tvm.nd.empty(out.shape, out.dtype) for out in symbol_table.outputs
            ]
            tvm_func(*inputs, *weights, *outputs)
            len(outputs)
            return tuple(paddle.to_tensor(out.asnumpy()) for out in outputs)

        return compiled_func

    def compile_placeholder(
        self,
        gl: paddlefx.GraphLayer,
        node: paddlefx.Node,
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor],
        inputs: list,
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
        self,
        gl: paddlefx.GraphLayer,
        node: paddlefx.Node,
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor],
        inputs: list,
    ):
        from tvm import te, topi

        module = gl
        names = node.target.split(".")
        while len(names) > 0:
            module = getattr(module, names.pop(0))
        if isinstance(module, nn.Linear):
            # TODO: pre-load weight and bias
            symbol_table.add_weight(
                f"{node.name}_weight",
                (
                    te.placeholder(
                        module.weight.T.shape,
                        paddle_dtype_to_str(module.weight.dtype),
                        name=f"params_{node.name}_weight",
                    ),
                    module.weight.T,
                ),
            )
            symbol_table.add_weight(
                f"{node.name}_bias",
                (
                    te.placeholder(
                        module.bias.shape,
                        paddle_dtype_to_str(module.bias.dtype),
                        name=f"params_{node.name}_bias",
                    ),
                    module.bias,
                ),
            )
            symbol_table[node.name] = topi.nn.dense(  # type: ignore
                symbol_table[str(node.args[0])],
                symbol_table[f"{node.name}_weight"],
                symbol_table[f"{node.name}_bias"],
            )
        elif isinstance(module, nn.Conv2D):
            symbol_table.add_weight(
                f"{node.name}_weight",
                (
                    te.placeholder(
                        module.weight.shape,
                        paddle_dtype_to_str(module.weight.dtype),
                        name=f"params_{node.name}_weight",
                    ),
                    module.weight,
                ),
            )

            if module.bias is not None:
                bias = module.bias.reshape((1, -1, 1, 1))
                symbol_table.add_weight(
                    f"{node.name}_bias",
                    (
                        te.placeholder(
                            bias.shape,
                            paddle_dtype_to_str(bias.dtype),
                            name=f"params_{node.name}_bias",
                        ),
                        bias,
                    ),
                )
                symbol_table[node.name] = topi.add(
                    topi.nn.conv2d(
                        symbol_table[str(node.args[0])],
                        symbol_table[f"{node.name}_weight"],
                        module._stride,
                        module._updated_padding,
                        module._dilation,
                    ),
                    symbol_table[f"{node.name}_bias"],
                )
            else:
                symbol_table[node.name] = topi.nn.conv2d(  # type: ignore
                    symbol_table[str(node.args[0])],
                    symbol_table[f"{node.name}_weight"],
                    module._stride,
                    module._updated_padding,
                    module._dilation,
                )
        else:
            raise CompilerError(f"Unsupported module: {module.__class__.__name__}")

    def compile_call_function(
        self,
        gl: paddlefx.GraphLayer,
        node: paddlefx.Node,
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor],
        inputs: list,
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
        self,
        gl: paddlefx.GraphLayer,
        node: paddlefx.Node,
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor],
        inputs: list,
    ):
        symbol_table.outputs = tuple(symbol_table[str(arg)] for arg in node.args[0])
