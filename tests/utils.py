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
    if isinstance(out, tuple):
        for i in range(len(res)):
            np.testing.assert_allclose(res[i], out[i], rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_allclose(res, out, rtol=1e-5, atol=1e-6)
