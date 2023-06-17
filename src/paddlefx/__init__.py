from __future__ import annotations

from .eval_frame import optimize  # noqa
from .graph import Graph  # noqa
from .graph_layer import GraphLayer  # noqa
from .interpreter import Interpreter  # noqa
from .node import Node  # noqa
from .symbolic_trace import Tracer, symbolic_trace  # noqa

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = version = '0.0.0.unknown'
    __version_tuple__ = version_tuple = (0, 0, 0, "unknown")
