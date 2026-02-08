"""
Type annotations for the simplified pipeline API.

These types make data flow explicit and visible at the point of use:
- Pipe[T]: Marks a parameter as receiving pipeline data from the previous step
- Ctx[T]: Marks a parameter as receiving pipeline context (user, dataset, etc.)
- Cfg[T]: Marks a parameter as receiving injected configuration
- Batch[T]: Type hint indicating batch processing semantics
- Stream[T]: Type hint indicating streaming/generator processing
- Drop: Sentinel value to filter items out of the pipeline
"""

from typing import Annotated, Any, Optional, TypeVar
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


class _CfgMarker:
    """Marker: this parameter receives injected configuration."""

    def __init__(self, config_key: str = ""):
        self.config_key = config_key

    def __repr__(self):
        return f"Cfg({self.config_key!r})" if self.config_key else "Cfg"


# Type aliases using Annotated
# Usage: def my_task(data: Pipe[list[str]]) -> list[str]: ...
Pipe = Annotated[T, _PipeMarker()]
Ctx = Annotated[T, _CtxMarker()]
Cfg = Annotated[T, _CfgMarker()]


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


class Batch(list):
    """Type hint indicating batch processing semantics.

    Example:
        async def process(items: Batch[dict]) -> Batch[dict]:
            return Batch([transform(x) for x in items])
    """

    pass


class Stream:
    """Type hint indicating streaming/generator processing.

    Example:
        async def process(items: Stream[dict]) -> Stream[dict]:
            for item in items:
                yield transform(item)
    """

    pass


class _Inject:
    """Dependency injection marker for special parameters.

    Example:
        @step
        async def my_step(
            data: dict,
            context: dict = inject("context"),
            user: User = inject("user"),
        ):
            return data
    """

    def __init__(self, param_name: str):
        self.param_name = param_name

    def __repr__(self):
        return f"inject({self.param_name!r})"


def inject(param_name: str) -> Any:
    """Dependency injection marker for special parameters.

    Usage:
        @step
        async def my_step(data: dict, user: User = inject("user")):
            return data
    """
    return _Inject(param_name)


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


def get_cfg_param_name(sig: inspect.Signature) -> Optional[str]:
    """Find the parameter annotated with Cfg[T] in a function signature."""
    for name, param in sig.parameters.items():
        for marker in _get_annotated_markers(param.annotation):
            if isinstance(marker, _CfgMarker):
                return name
    return None
