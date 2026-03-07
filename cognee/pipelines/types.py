"""
Type annotations for the simplified pipeline API.

- Pipe[T]: Marks a parameter as receiving pipeline data from the previous step
- Ctx[T]: Marks a parameter as receiving pipeline context (user, dataset, etc.)
- Drop: Sentinel value to filter items out of the pipeline
"""

from typing import Annotated, Optional, TypeVar
import inspect

T = TypeVar("T")


class _PipeMarker:
    """Marker: this parameter receives pipeline data from the previous step."""

    def __repr__(self):
        return "Pipe"


class _CtxMarker:
    """Marker: this parameter receives pipeline context."""

    def __repr__(self):
        return "Ctx"


# Type aliases using Annotated
# Usage: def my_task(data: Pipe[list[str]]) -> list[str]: ...
Pipe = Annotated[T, _PipeMarker()]
Ctx = Annotated[T, _CtxMarker()]


class _Drop:
    """Sentinel value: return Drop from a step to filter an item out of the pipeline.

    Example:
        async def filter_short(text: str) -> str:
            if len(text) < 10:
                return Drop  # This item is removed from the pipeline
            return text
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Drop"

    def __bool__(self):
        return False


Drop = _Drop()


# --- Introspection helpers ---


def _get_annotated_markers(annotation):
    """Extract markers from an Annotated type."""
    if hasattr(annotation, "__metadata__"):
        return annotation.__metadata__
    return ()


def get_pipe_param_name(sig: inspect.Signature) -> Optional[str]:
    """Find the parameter annotated with Pipe[T] in a function signature."""
    for name, param in sig.parameters.items():
        for marker in _get_annotated_markers(param.annotation):
            if isinstance(marker, _PipeMarker):
                return name
    return None


def get_ctx_param_name(sig: inspect.Signature) -> Optional[str]:
    """Find the parameter annotated with Ctx[T] in a function signature."""
    for name, param in sig.parameters.items():
        for marker in _get_annotated_markers(param.annotation):
            if isinstance(marker, _CtxMarker):
                return name
    return None
