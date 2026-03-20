"""
Type annotations for the simplified pipeline API.

- Pipe[T]: Marks a parameter as receiving pipeline data from the previous step
- Drop: Sentinel value to filter items out of the pipeline
"""

from typing import Annotated, Optional, TypeVar
import inspect

T = TypeVar("T")


class _PipeMarker:
    """Marker: this parameter receives pipeline data from the previous step."""

    def __repr__(self):
        return "Pipe"


# Type alias using Annotated
# Usage: def my_task(data: Pipe[list[str]]) -> list[str]: ...
Pipe = Annotated[T, _PipeMarker()]


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


def inject_context_kwargs(sig: inspect.Signature, context, num_positional_args: int = 1) -> dict:
    """Determine kwargs to inject based on function signature and context.

    Inspects the function signature and injects matching context values by
    parameter name. If the function declares parameters like `user`, `dataset`,
    etc., and those keys exist in the context dict, they are injected directly.

    Parameters that are already being filled by positional arguments are
    skipped to avoid "multiple values" errors.

    For backward compatibility, if the function has a parameter named `context`
    and `context` is not itself a key in the context dict, the full context
    value is injected.

    Args:
        sig: The function's inspect.Signature.
        context: The context value (usually a dict, but can be any type for
                 legacy compatibility).
        num_positional_args: Number of positional arguments already being passed
                            to the function. These parameters are skipped.
                            Default 1 (the pipeline data argument).

    Returns:
        Dict of kwargs to inject into the function call.
    """
    kwargs = {}
    if context is None:
        return kwargs

    # Determine which parameter names are being filled by positional args.
    # These must be skipped to avoid "got multiple values" errors.
    params = list(sig.parameters.values())
    positional_names = set()
    count = 0
    for param in params:
        if count >= num_positional_args:
            break
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_names.add(param.name)
            count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            break

    if isinstance(context, dict):
        for param_name in sig.parameters:
            if param_name in positional_names:
                continue
            if param_name in context:
                kwargs[param_name] = context[param_name]
        # Legacy: if function has a 'context' param and it's not a dict key,
        # inject the full context dict
        if (
            "context" in sig.parameters
            and "context" not in kwargs
            and "context" not in positional_names
        ):
            kwargs["context"] = context
    else:
        # Non-dict context (e.g. context=7 in legacy tests)
        if "context" in sig.parameters and "context" not in positional_names:
            kwargs["context"] = context

    return kwargs
