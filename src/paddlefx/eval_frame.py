from __future__ import annotations

import functools
import logging
import types

from ._eval_frame import set_eval_frame
from .convert_frame import convert_frame


class DynamoContext:
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

            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(old_callback)

        _fn.raw_fn = fn

        return _fn


class DisableContext(DynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def disable(fn=None):
    return DisableContext()(fn)


def optimize(backend: callable):
    def _fn(compiler_fn):
        inner_convert = convert_frame(compiler_fn)

        def __fn(frame: types.FrameType):
            try:
                result = inner_convert(frame)
                return result
            except NotImplementedError as e:
                logging.warning(f"NotImplementedError: {e}")
            except Exception:
                raise
            return None

        return __fn

    return DynamoContext(_fn(backend))
