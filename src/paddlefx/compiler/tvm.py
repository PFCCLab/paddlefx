from __future__ import annotations

from typing import Callable

import paddle
import paddle.device
import tvm

from tvm import auto_scheduler, relay

import paddlefx

from .base import CompilerBase


class TVMCompiler(CompilerBase):
    def __init__(
        self,
        *,
        allow_fallback: bool = False,
        full_graph: bool = False,
        print_tabular_mode: str | None = None,
        target: str | tvm.target.Target = "llvm",
        tune_mode: str = "auto_scheduler",
    ):
        import tvm

        super().__init__(
            allow_fallback=allow_fallback,
            full_graph=full_graph,
            print_tabular_mode=print_tabular_mode,
        )

        self.target = tvm.target.Target(target)
        self.device = tvm.device(self.target.kind.name)
        self.tune_mode = tune_mode

    def compile(self, gl: paddlefx.GraphLayer, example_inputs: list) -> Callable:
        shape_dict = {}
        for node in gl.graph.nodes:
            if node.op == "placeholder":
                shape_dict[node.name] = example_inputs[self.input_index].shape
                self.input_index += 1
        gl(*example_inputs)
        model_path = ".model"
        paddle.jit.save(gl, model_path)
        translated_layer = paddle.jit.load(model_path)
        mod, params = relay.frontend.from_paddle(translated_layer)

        if self.tune_mode == "auto_scheduler":
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"], params, self.target
            )

            for idx, task in enumerate(tasks):
                print(
                    "========== Task %d  (workload key: %s) =========="
                    % (idx, task.workload_key)
                )
                print(task.compute_dag)

            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=200,
                early_stopping=10,
                measure_callbacks=[auto_scheduler.RecordToFile("log_file")],
            )

            tuner.tune(tune_option)
            with auto_scheduler.ApplyHistoryBest("log_file"):
                with tvm.transform.PassContext(
                    opt_level=3, config={"relay.backend.use_auto_scheduler": True}
                ):
                    executor = relay.build_module.create_executor(
                        "graph", mod, self.device, self.target, params
                    ).evaluate()

        else:
            with tvm.transform.PassContext(opt_level=3):
                executor = relay.build_module.create_executor(
                    "graph", mod, self.device, self.target, params
                ).evaluate()

        def compiled_func(*args):
            inputs = [tvm.nd.array(arg.numpy(), device=self.device) for arg in args]
            tvm_output = executor(*inputs)
            if isinstance(tvm_output, (list, tuple)):
                return tuple(paddle.to_tensor(out.asnumpy()) for out in tvm_output)
            else:
                return (paddle.to_tensor(tvm_output.asnumpy()),)

        return compiled_func
