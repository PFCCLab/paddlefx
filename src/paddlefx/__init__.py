try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = version = '0.0.0.unknown'
    __version_tuple__ = version_tuple = (0, 0, 0, "unknown")


from .symbolic_trace import symbolic_trace, Tracer
from .graph import Graph
from .node import Node
from .proxy import Proxy