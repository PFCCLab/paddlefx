from __future__ import annotations

from typing import Callable

import numpy as np

import paddlefx


def check_func(func, *args, backend: Callable | None = None):
    if backend is None:
        comiled_func = paddlefx.optimize(func)
    else:
        comiled_func = paddlefx.optimize(func, backend=backend)
    out = func(*args)
    res = comiled_func(*args)
    np.testing.assert_allclose(res, out)
