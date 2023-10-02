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


def auto_scheduler(symbol_table, target):
    log_file = f"{hash(tuple(out.name for out in symbol_table.outputs))}.json"
    task = auto_scheduler.SearchTask(
        func=auto_scheduler.register_workload(lambda: symbol_table.outputs),
        target=target,
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,  # change this to 1000 to achieve the best performance
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        # verbose=2,
    )
    task.tune(tune_option)
    # Apply the best schedule
    schedule, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(schedule, args, simple_mode=True))


class TVMCompiler(CompilerBase):
    def gen_compiled_func(self, symbol_table: SymbolTable[te.Tensor, paddle.Tensor]):
        import tvm

        from tvm import te

        device = paddle.device.get_device()
        # device = "gpu"
        if device == "cpu":
            target = tvm.target.Target(target="llvm")
            dev = tvm.cpu()
        elif device == "gpu":
            target = tvm.target.Target(target="cuda", host="llvm")
            dev = tvm.cuda()
        else:
            raise CompilerError(f"Unsupported device in tvm backend: {device}")
        target = tvm.target.Target(target="llvm -mtriple=x86_64-linux-gnu")

        schedule = te.create_schedule([out.op for out in symbol_table.outputs])

        print("building tvm func")
        tvm_func = tvm.build(
            schedule,
            [*symbol_table.inputs, *symbol_table.outputs],
            target,
            name="tvm_func",
        )
        print("builded tvm func")

        weights = [tvm.nd.array(p.numpy(), device=dev) for p in symbol_table.weights]

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy(), device=dev) for arg in args]
            outputs = [
                tvm.nd.empty(out.shape, out.dtype, device=dev)
                for out in symbol_table.outputs
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
                        module._data_format,
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
        elif isinstance(module, nn.BatchNorm2D):
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
            symbol_table.add_weight(
                f"{node.name}_mean",
                (
                    te.placeholder(
                        module._mean.shape,
                        paddle_dtype_to_str(module._mean.dtype),
                        name=f"params_{node.name}_mean",
                    ),
                    module._mean,
                ),
            )
            symbol_table.add_weight(
                f"{node.name}_variance",
                (
                    te.placeholder(
                        module._variance.shape,
                        paddle_dtype_to_str(module._variance.dtype),
                        name=f"params_{node.name}_variance",
                    ),
                    module._variance,
                ),
            )
            symbol_table[node.name] = topi.nn.batch_norm(
                symbol_table[str(node.args[0])],
                symbol_table[f"{node.name}_weight"],
                symbol_table[f"{node.name}_bias"],
                symbol_table[f"{node.name}_mean"],
                symbol_table[f"{node.name}_variance"],
                epsilon=module._epsilon,
                training=module.training,
            )[0]
        elif isinstance(module, nn.ReLU):
            symbol_table[node.name] = topi.nn.relu(symbol_table[str(node.args[0])])  # type: ignore
        elif isinstance(module, nn.MaxPool2D):
            symbol_table[node.name] = topi.nn.pool2d(
                symbol_table[str(node.args[0])],
                [module.ksize, module.ksize]
                if isinstance(module.ksize, int)
                else module.ksize,
                [module.stride, module.stride]
                if isinstance(module.stride, int)
                else module.stride,
                [1, 1],
                [module.padding, module.padding, module.padding, module.padding]
                if isinstance(module.padding, int)
                else module.padding,
                "max",
                module.ceil_mode,
            )
        elif isinstance(module, nn.AdaptiveAvgPool2D):
            symbol_table[node.name] = topi.nn.adaptive_pool(
                symbol_table[str(node.args[0])],
                module._output_size,
                "avg",
            )
        elif isinstance(module, nn.AvgPool2D):
            pass
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
            "iadd": topi.add,
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
        elif target_name == "flatten":
            inp = symbol_table[str(node.args[0])]
            batch = inp.shape[0]
            from functools import reduce

            shape = [batch, reduce(lambda x, y: x * y, inp.shape[1:])]
            symbol_table[node.name] = topi.reshape(
                inp,
                shape,
            )
        else:
            raise CompilerError(f"Unsupported function: {target_name}")

    def compile_output(
        self,
        gl: paddlefx.GraphLayer,
        node: paddlefx.Node,
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor],
        inputs: list,
    ):
        symbol_table.outputs = tuple(symbol_table[str(arg)] for arg in node.args[0])
