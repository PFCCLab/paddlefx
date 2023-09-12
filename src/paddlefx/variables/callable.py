from __future__ import annotations

import inspect
import operator
import types

from typing import TYPE_CHECKING, Any, Callable

import paddle

from .base import ObjectVariable, VariableBase

if TYPE_CHECKING:
    from ..pyeval import PyEvalBase


class CallableVariable(VariableBase):
    def __init__(self, fn: Callable):
        super().__init__(var=fn, vtype=type(fn))
        self.fn = fn

    def __str__(self):
        if self.fn is None:
            name = "None"
        else:
            if hasattr(self.fn, __name__):
                name = self.fn.__name__
            else:
                name = repr(self.fn)
        return f"{self.__class__.__name__}({name})"

    def __call__(self, tx: PyEvalBase, *args: VariableBase, **kwargs) -> Any:
        # TODO: better org
        fn = self.fn
        graph = tx.output.graph

        # nn.layer
        if isinstance(fn, paddle.nn.Layer):
            # unroll nn.Sequential
            if 'container' in fn.__module__:
                assert not kwargs
                (arg,) = args
                for idx, submod in enumerate(fn):
                    tx.call_function(
                        CallableVariable(fn=submod),
                        [arg],
                        {},
                    )
                    arg = tx.stack.pop()
                return arg
            elif not fn.__module__.startswith('paddle.nn'):
                globals()['self'] = tx.f_locals['self']
                result = tx.inline_call_function(
                    CallableVariable(fn=fn.forward.__func__), (self, *args), kwargs
                )
                del globals()['self']
                return result
            else:
                # basic layer
                ot = args[0].vtype
                target = ''
                model = (
                    tx.f_locals['self'] if 'self' in tx.f_locals else globals()['self']
                )
                for name, layers in model.named_sublayers():
                    if fn is layers:
                        target = name
                        break
                return VariableBase(
                    vtype=ot, node=graph.call_module(target, args, kwargs)
                )
        elif fn.__module__.startswith("paddle"):
            # TODO: support multiple ouputs and containers
            ot = args[0].vtype
            output = graph.call_function(fn, args, kwargs, ot)
            return VariableBase(vtype=ot, node=output)
        elif inspect.isbuiltin(fn):
            if fn is print:
                raise NotImplementedError("print() is not supported")
            elif fn is getattr:
                object, name = args
                attr = getattr(object.var, name.var)
                if callable(attr):
                    if isinstance(attr, types.MethodType):
                        ot = args[0].vtype
                        return MethodVariable(fn=attr, vtype=ot)
                    else:
                        # the attr could be callable function
                        return CallableVariable(fn=attr)
                else:
                    return VariableBase(var=attr)
            elif fn in [operator.add, operator.sub, operator.iadd]:
                ot = args[0].vtype
                output = graph.call_function(fn, args, kwargs, ot)
                return VariableBase(vtype=ot, node=output)
            elif fn in [operator.gt]:
                ot = args[0].vtype
                output = graph.call_function(fn, args, kwargs, ot)
                return VariableBase(vtype=ot, node=output)
            elif fn in [operator.is_, operator.is_not]:
                ot = args[0].vtype
                output = graph.call_function(fn, args, kwargs, ot)
                return VariableBase(vtype=ot, node=output)
            else:
                raise NotImplementedError(f"builtin {fn} is not supported")

        return tx.inline_call_function(self, args, kwargs)

    def call_function(
        self,
        translator,
        args: list[VariableBase],
        kwargs: dict[str, VariableBase],
    ) -> VariableBase:
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)
        fn_name = self.fn.__name__
        handler = getattr(self, f"call_{fn_name}", None)
        if handler:
            return handler(translator, *args, **kwargs)
        return ObjectVariable(
            translator.output.create_node('call_function', self.fn, args, kwargs)
        )
        # raise NotImplementedError(f"{fn_name} is not implemented now")


class PaddleVariable(CallableVariable):
    pass


# note: python module
class ModuleVariable(ObjectVariable):
    def __init__(self, module: types.ModuleType):
        super().__init__(module)

    def __str__(self):
        return self.obj.__name__

    def __getattr__(self, attr: str):
        out_obj = getattr(self.obj, attr)
        if isinstance(out_obj, types.ModuleType):
            return ModuleVariable(out_obj)
        elif isinstance(out_obj, types.FunctionType):
            return CallableVariable(out_obj)
        else:
            return ObjectVariable(out_obj)


class PaddleLayerVariable(CallableVariable):
    def __init__(self, fn):
        super().__init__(fn)


class MethodVariable(VariableBase):
    def __init__(self, fn, vtype=None):
        super().__init__(var=fn, vtype=type(fn) if vtype is None else vtype)
        self.fn = fn

    def __call__(self, tx: PyEvalBase, *args: VariableBase, **kwargs):
        fn = self.fn
        graph = tx.output.graph
        ot = self.vtype
        output = graph.call_method(fn, args, kwargs, ot)
        return VariableBase(vtype=ot, node=output)
