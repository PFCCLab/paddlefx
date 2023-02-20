from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict
import torch

if TYPE_CHECKING:
    from .graph import Graph


BaseArgumentTypes = Union[str, int, float, bool, torch.dtype, torch.Tensor]
base_types = BaseArgumentTypes.__args__  # type: ignore

Target = Union[Callable[..., Any], str]

Argument = Optional[Union[
    Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
    List[Any],  # actually Argument
    Dict[str, Any],  # actually Argument
    slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
    'Node',
    BaseArgumentTypes
]]