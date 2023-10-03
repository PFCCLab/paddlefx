from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Callable

import paddle
import paddle.device

from paddle import nn

import paddlefx

from .base import CompilerBase, CompilerError, SymbolTable, paddle_dtype_to_str

if TYPE_CHECKING:
    import tvm

    from tvm import te


def auto_tunning(symbol_table, target, workload):
    from tvm import auto_scheduler

    log_file = f"{hash(tuple(out.name for out in symbol_table.outputs))}.json"
    task = auto_scheduler.SearchTask(
        func=auto_scheduler.register_workload(workload),
        args=(1,),
        target=target,
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=1,
    )
    task.tune(tune_option)
    # Apply the best schedule
    schedule, args = task.apply_best(log_file)
    return schedule, args


class TVMCompiler(CompilerBase):
    def __init__(
        self,
        *,
        allow_fallback: bool = False,
        full_graph: bool = False,
        print_tabular_mode: str | None = None,
        target: str | tvm.target.Target = "llvm",
    ):
        import tvm

        super().__init__(
            allow_fallback=allow_fallback,
            full_graph=full_graph,
            print_tabular_mode=print_tabular_mode,
        )

        self.target = tvm.target.Target(target) if isinstance(target, str) else target
        self.target = tvm.target.Target(target="llvm -mtriple=x86_64-linux-gnu")
        # self.target = tvm.target.Target(target="cuda", host="llvm")

    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        symbol_table = self.gen_symbol_table(gl, example_inputs)
        return self.gen_compiled_func(symbol_table)

    def gen_symbol_table(
        self, gl: paddlefx.GraphLayer, example_inputs: list
    ) -> SymbolTable[te.Tensor, paddle.Tensor]:
        symbol_table: SymbolTable[te.Tensor, paddle.Tensor] = SymbolTable()
        for node in gl.graph.nodes:
            getattr(self, f"compile_{node.op}")(gl, node, symbol_table, example_inputs)
        return symbol_table

    def gen_compiled_func(self, symbol_table: SymbolTable[te.Tensor, paddle.Tensor]):
        import tvm

        from tvm import te

        device = tvm.device(self.target.kind.name, 0)
        if self.target.kind.name == "llvm":
            # func = te.create_prim_func(symbol_table.all_symbols)
            # ir_module_from_te = IRModule({"main": func})
            # tasks, task_weights = auto_scheduler.extract_tasks(ir_module_from_te, None, self.target)

            # for idx, task in enumerate(tasks):
            #     print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            #     print(task.compute_dag)

            schedule = te.create_schedule([out.op for out in symbol_table.outputs])

            args = list(symbol_table.all_symbols)

            for out in symbol_table.outputs:
                # print(schedule[out].op.axis, len(schedule[out].op.axis))
                # print(schedule[out].op.reduce_axis, len(schedule[out].op.reduce_axis))
                # fuse_axis = schedule[out].fuse(*schedule[out].op.axis)
                # bx, tx = schedule[out].split(fuse_axis, factor=64)

                schedule[out].unroll(schedule[out].op.axis[0])
                if len(schedule[out].op.axis) >= 2:
                    schedule[out].parallel(schedule[out].op.axis[1])
                if len(schedule[out].op.axis) >= 3:
                    schedule[out].vectorize(schedule[out].op.axis[2])

                print(tvm.lower(schedule, args, simple_mode=True))

        elif self.target.kind.name == "cuda":
            # schedule, args = auto_tunning(symbol_table, self.target)
            schedule = te.create_schedule([out.op for out in symbol_table.outputs])
            args = list(symbol_table.all_symbols)
            for name, out in symbol_table:
                if isinstance(out.op, te.PlaceholderOp):
                    continue
                fused = schedule[out].fuse(*schedule[out].op.axis)
                bx, tx = schedule[out].split(fused, factor=64)
                schedule[out].bind(bx, te.thread_axis("blockIdx.x"))
                schedule[out].bind(tx, te.thread_axis("threadIdx.x"))
            raise CompilerError("cuda is not supported yet")
        else:
            raise CompilerError(f"{self.target.kind.name} is not supported yet")

        tvm_func = tvm.build(
            schedule,
            args,
            self.target,
            name="tvm_func",
        )
        print("builded tvm func")

        params = [
            tvm.nd.array(p[1].numpy(), device=device) for p in symbol_table.params
        ]

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy(), device=device) for arg in args]
            outputs = [
                tvm.nd.empty(out.shape, out.dtype, device=device)
                for out in symbol_table.outputs
            ]
            tvm_func(*inputs, *params, *outputs)
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
            symbol_table.add_param(
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
            symbol_table.add_param(
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
            symbol_table.add_param(
                f"{node.name}_weight",
                (
                    te.placeholder(
                        module.weight.shape,
                        paddle_dtype_to_str(module.weight.dtype),
                    ),
                    module.weight,
                ),
            )

            if module.bias is not None:
                bias = module.bias.reshape((1, -1, 1, 1))
                symbol_table.add_param(
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
                symbol_table[f"temp_{node.name}_conv2d"] = topi.nn.conv2d(  # type: ignore
                    symbol_table[str(node.args[0])],
                    symbol_table[f"{node.name}_weight"],
                    module._stride,
                    module._updated_padding,
                    module._dilation,
                    module._data_format,
                )
                symbol_table[node.name] = topi.add(
                    symbol_table[f"temp_{node.name}_conv2d"],
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
            symbol_table.add_param(
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
            symbol_table.add_param(
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
            symbol_table.add_param(
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
            symbol_table.add_param(
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
