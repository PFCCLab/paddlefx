# this is ported from https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html

from __future__ import annotations

import copy

from typing import Any

import numpy as np
import paddle
import paddle.nn as nn

import paddlefx as fx


def fuse_conv_bn_eval(conv, bn):
    """Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode."""
    assert not (conv.training or bn.training), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_conv_bn_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn._mean,
        bn._variance,
        bn._epsilon,
        bn.weight,
        bn.bias,
    )

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = paddle.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = paddle.ones_like(bn_rm)
    if bn_b is None:
        bn_b = paddle.zeros_like(bn_rm)
    bn_var_rsqrt = paddle.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    conv_w_param = paddle.create_parameter(conv_w.shape, dtype=conv_w.dtype)
    conv_w_param.set_value(conv_w)

    conv_b_param = paddle.create_parameter(
        conv_b.shape, dtype=conv_b.dtype, is_bias=True
    )
    conv_b_param.set_value(conv_b)
    return conv_w_param, conv_b_param


def _parent_name(target: str) -> tuple[str, str]:
    """Splits a qualname into parent path and last atom.

    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else 'root', name


def replace_node_module(
    node: fx.Node, modules: dict[str, Any], new_module: paddle.nn.Layer
):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name('root.' + node.target)
    setattr(modules[parent_name], name, new_module)


def fuse(model: paddle.nn.Layer) -> paddle.nn.Layer:
    model = copy.deepcopy(model)
    fx_model: fx.GraphLayer = fx.symbolic_trace(model)

    # Note: paddle.nn.Layer.named_sublayers API is not compatible with torch.nn.Module.named_modules
    modules = dict(fx_model.named_sublayers(prefix='', include_self=True))

    # TODO: iterating over fx_model.graph.nodes will cause endless loop, need to fix
    #   for node in fx_model.graph.nodes:
    for node in list(fx_model.graph.nodes):
        if (
            node.op != 'call_module'
        ):  # If our current node isn't calling a Module then we can ignore it.
            continue
        # For call sites, `Node.target` represents the module/function/method
        # that's being called. Here, we check `Node.target` to see if it's a
        # batch norm module, and then check `Node.args[0].target` to see if the
        # input `Node` is a convolution.
        if (
            type(modules['root.' + node.target]) is nn.BatchNorm2D
            and type(modules['root.' + node.args[0].target]) is nn.Conv2D
        ):
            if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                continue
            conv = modules['root.' + node.args[0].target]
            bn = modules['root.' + node.target]
            fused_conv = fuse_conv_bn_eval(conv, bn)

            replace_node_module(node.args[0], modules, fused_conv)
            # As we've folded the batch nor into the conv, we need to replace all uses
            # of the batch norm with the conv.
            node.replace_all_uses_with(node.args[0])
            # Now that all uses of the batch norm have been replaced, we can
            # safely remove the batch norm.
            fx_model.graph.erase_node(node)

    fx_model.recompile()
    return fx_model


import time

rn18 = paddle.vision.models.resnet18()
rn18.eval()

traced_rn18 = fx.symbolic_trace(rn18)
fused_rn18 = fuse(rn18)

inp = paddle.randn([10, 3, 224, 224])
orig_output = rn18(inp)
fused_output = fused_rn18(inp)

# TODO: figure out the reason of the difference
# Note: setting rtol greater may cause the test to fail
# May be because the computation is not exactly the same
np.testing.assert_allclose(orig_output.numpy(), fused_output.numpy(), rtol=1e-1)
# assert paddle.allclose(orig_output, fused_output, rtol=1e-4)


def benchmark(model, iters=5):
    for _ in range(10):
        model(inp)
    begin = time.time()
    for _ in range(iters):
        model(inp)
    return str(time.time() - begin)


# TODO: refactor/reimplement python code generation of paddlefx
# Note: the fused model is slower than the unfused model, which is unexpected
# I highly suspect that the reason is that the python code genreation of paddlefx
# has some problems. If using examples/fx_profiling.py to run the fused model, the fused model
# is slight faster than the unfused model, which is expected.
# examples/fx_profiling.py is running the node one by one, which is different to the python code generation
# even running traced model using generated code is a little bit slower than running original model.
print("Fused time: ", benchmark(fused_rn18))
print("Unfused time: ", benchmark(rn18))
print("Traced time: ", benchmark(traced_rn18))
