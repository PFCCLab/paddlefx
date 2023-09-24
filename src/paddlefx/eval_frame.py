from __future__ import annotations

import functools
import logging
import types

from typing import Callable

from ._eval_frame import set_eval_frame
from .compiler import mlir_compiler
from .convert_frame import convert_frame


class BaseContext:
    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        self.old_callback = set_eval_frame(self.callback)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        set_eval_frame(self.old_callback)

    def __call__(self, fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            old_callback = set_eval_frame(self.callback)

            result = fn(*args, **kwargs)
            set_eval_frame(old_callback)
            return result

        _fn.wrapped_fn = fn  # type: ignore

        return _fn


class DisableContext(BaseContext):
    def __init__(self):
        super().__init__(callback=None)


def disable(fn=None):
    return DisableContext()(fn)


def optimize(backend: Callable = mlir_compiler):
    def _fn(backend: Callable):
        def __fn(frame: types.FrameType):
            try:
                result = convert_frame(frame, backend)
                return result
            except NotImplementedError as e:
                logging.debug(f"!! NotImplementedError: {e}")
            except Exception:
                raise
            return None

        return __fn

    return BaseContext(_fn(backend))
