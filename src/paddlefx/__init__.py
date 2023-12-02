from .eval_frame import optimize
from .graph import Graph
from .graph_layer import GraphLayer
from .graph_viewer import FxGraphViewer
from .interpreter import Interpreter
from .node import Node
from .symbolic_trace import Tracer, symbolic_trace

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = version = "0.0.0.unknown"
    __version_tuple__ = version_tuple = (0, 0, 0, "unknown")
