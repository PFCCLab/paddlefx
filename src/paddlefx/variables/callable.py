from __future__ import annotations

import inspect
import operator
import types

from typing import TYPE_CHECKING, Any, Callable

import paddle

from ..source import Source
from ..utils import BreakGraphError
from .base import ObjectVariable, TensorVariable, VariableBase

if TYPE_CHECKING:
    from ..pyeval import PyEvalBase


class CallableVariable(VariableBase):
    def __init__(
        self,
        fn: Callable,
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(var=fn, tx=tx, source=source, node=node)

    def __str__(self):
        if self.var is None:
            name = "None"
        else:
            if hasattr(self.var, __name__):
                name = self.var.__name__
            else:
                name = repr(self.var)
        return f"{self.__class__.__name__}({name})"

    def __call__(self, tx: PyEvalBase, *args: VariableBase, **kwargs) -> VariableBase:
        # TODO: better org
        fn = self.var
        graph = tx.output.graph

        # nn.layer
        if isinstance(fn, paddle.nn.Layer):  # type: ignore
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
                    CallableVariable(fn=fn.forward.__func__, tx=tx),
                    (self, *args),
                    kwargs,
                )
                del globals()['self']
                return result
            else:  # basic layer
                ot = type(args[0].var)

                target = ''
                model = (
                    tx.f_locals['self'] if 'self' in tx.f_locals else globals()['self']
                )
                for name, layers in model.named_sublayers():
                    if fn is layers:
                        target = name
                        break
                return TensorVariable(
                    None, node=graph.call_module(target, args, kwargs)
                )
        elif fn.__module__.startswith("paddle"):
            # TODO: support multiple ouputs and containers
            ot = type(args[0].var)

            output = graph.call_function(fn, args, kwargs, ot)
            return TensorVariable(None, node=output)
        elif inspect.isbuiltin(fn):
            if fn is print:
                raise BreakGraphError("print is triggered")
            elif fn is getattr:
                object, name = args
                attr = getattr(object.var, name.var)
                if callable(attr):
                    if isinstance(attr, types.MethodType):
                        # For method variables
                        ot = type(args[0].var)

                        return CallableVariable(fn, tx=tx)
                    else:
                        # the attr could be callable function
                        return CallableVariable(fn, tx=tx)
                else:
                    return VariableBase(var=attr)
            elif fn in [
                operator.add,
                operator.sub,
                operator.mul,
                operator.truediv,
                operator.iadd,
            ]:
                ot = type(args[0].var)

                output = graph.call_function(fn, args, kwargs, ot)
                return TensorVariable(None, node=output)
            elif fn in [operator.gt, operator.lt, operator.ge, operator.le]:
                ot = type(args[0].var)

                output = graph.call_function(fn, args, kwargs, ot)
                return TensorVariable(None, node=output)
            elif fn in [operator.is_, operator.is_not]:
                ot = type(args[0].var)

                output = graph.call_function(fn, args, kwargs, ot)
                return TensorVariable(None, node=output)
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
        fn_name = self.var.__name__
        handler = getattr(self, f"call_{fn_name}", None)
        if handler:
            return handler(translator, *args, **kwargs)
        return ObjectVariable(
            translator.output.create_node('call_function', self.var, args, kwargs)
        )
        # raise NotImplementedError(f"{fn_name} is not implemented now")


class PaddleVariable(CallableVariable):
    pass


# note: python module
class ModuleVariable(ObjectVariable):
    def __init__(
        self,
        module: types.ModuleType,
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(module, tx=tx, source=source, node=node)

    def __getattr__(self, attr: str):
        out_obj = getattr(self.obj, attr)
        if isinstance(out_obj, types.ModuleType):
            return ModuleVariable(out_obj)
        elif isinstance(out_obj, types.FunctionType):
            return CallableVariable(out_obj)
        else:
            return ObjectVariable(out_obj)


class PaddleLayerVariable(CallableVariable):
    def __init__(
        self,
        fn,
        *,
        tx: PyEvalBase | None = None,
        source: Source | None = None,
        node: Any = None,
    ):
        super().__init__(fn, tx=tx, source=source, node=node)


# class BuiltinVariable(CallableVariable):
#     def __init__(self, fn: Callable[..., Any]):
#         super().__init__(fn)

#     def call_function(self, /, *args, **kwargs):
#         # Lookup the handler from dispatcher
#         handler = Dispatcher.dispatch(self.value, *args, **kwargs)
#         if handler is not None:
#             return handler(*args, **kwargs)

#         # Try to inline call the magic function
#         magic_methods = magic_method_builtin_dispatch(self.value)
#         for magic_method in magic_methods:
#             sorted_args = args
#             if magic_method.is_reverse:
#                 sorted_args = sorted_args[::-1]
#             arg_type = sorted_args[0].get_py_type()
#             if hasattr(arg_type, magic_method.name):
#                 class_fn = getattr(arg_type, magic_method.name)
#                 class_var = VariableFactory.from_value(
#                     arg_type,
#                     self.graph,
#                     GetAttrTracker(args[0], "__class__"),
#                 )
#                 assert isinstance(class_var, VariableBase)
#                 fn_var = VariableFactory.from_value(
#                     class_fn,
#                     self.graph,
#                     GetAttrTracker(class_var, class_fn.__name__),
#                 )
#                 assert isinstance(fn_var, VariableBase)
#                 return fn_var(*args)

#         # Break graph if neither of the above conditions is met
#         arg_types = ", ".join([type(arg).__name__ for arg in args])
#         fn_name = self.value.__name__ if hasattr(self.value, '__name__') else self.value
#         raise BreakGraphError(
#             f"Not support builtin function: {fn_name} with args: Args({arg_types})"
#         )
