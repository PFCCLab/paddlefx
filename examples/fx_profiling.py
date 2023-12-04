# this is from: https://github.com/pytorch/tutorials/blob/main/intermediate_source/fx_profiling_tutorial.py

from __future__ import annotations

import statistics
import time

from typing import Any

import paddle
import paddle.nn
import tabulate

from paddle.vision.models import resnet18

import paddlefx

from paddlefx import Interpreter, symbolic_trace

net = resnet18()
example_input = paddle.rand([8, 3, 224, 224])

# traced_rn18 = symbolic_trace(net)
# traced_rn18.graph.print_tabular()


class ProfilingInterpreter(Interpreter):
    def __init__(self, mod: paddle.nn.Layer):
        gm = symbolic_trace(mod)
        super().__init__(gm)

        self.total_runtime_sec: list[float] = []
        self.runtimes_sec: dict[paddlefx.Node, list[float]] = {}

    def run(self, *args) -> Any:
        # Record the time we started running the model
        t_start = time.time()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        t_end = time.time()
        # Store the total elapsed time this model execution took in the
        # ProfilingInterpreter
        self.total_runtime_sec.append(t_end - t_start)
        return return_val

    def run_node(self, n: paddlefx.Node) -> Any:
        # Record the time we started running the op
        t_start = time.time()
        # Run the op by delegating back into Interpreter.run_node()
        return_val = super().run_node(n)
        # Record the time we finished running the op
        t_end = time.time()
        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(t_end - t_start)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort: bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries: list[list[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = statistics.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtime
            node_summaries.append([node.op, str(node), mean_runtime, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers: list[str] = [
            "Op type",
            "Op",
            "Average runtime (s)",
            "Pct total runtime",
        ]
        return tabulate.tabulate(node_summaries, headers=headers)


interp = ProfilingInterpreter(net)
for _ in range(10):
    interp.run(example_input)
print(interp.summary(True))
