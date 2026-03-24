from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import inspect
from typing import Any, Awaitable, Callable, TypeVar, cast

from .prompt_trace_context import AgentContextTrace

T = TypeVar("T")


_current_agent_context_trace: ContextVar[AgentContextTrace | None] = ContextVar(
    "_current_agent_context_trace", default=None
)


def get_current_agent_context_trace() -> AgentContextTrace | None:
    return _current_agent_context_trace.get()


def _start_agent_context_trace(origin_function: str, with_memory: bool) -> tuple[AgentContextTrace, object]:
    context_trace = AgentContextTrace(
        trace_metadata={
            "origin_function": origin_function,
            "with_memory": with_memory,
        }
    )
    token = _current_agent_context_trace.set(context_trace)
    return context_trace, token


def _decorate_agentic_root(
    func: Callable[..., T], *, with_memory: bool
) -> Callable[..., tuple[T, AgentContextTrace]]:
    """Wrap a function with AgentContextTrace lifecycle management."""

    if inspect.iscoroutinefunction(func):
        async_func = cast(Callable[..., Awaitable[T]], func)

        @wraps(async_func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> tuple[T, AgentContextTrace]:
            context_trace, token = _start_agent_context_trace(
                async_func.__name__, with_memory=with_memory
            )
            try:
                result = await async_func(*args, **kwargs)
                return result, context_trace
            finally:
                _current_agent_context_trace.reset(token)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> tuple[T, AgentContextTrace]:
        context_trace, token = _start_agent_context_trace(func.__name__, with_memory=with_memory)
        try:
            result = func(*args, **kwargs)
            return result, context_trace
        finally:
            _current_agent_context_trace.reset(token)

    return sync_wrapper


def agentic_trace_root(
    func: Callable[..., T] | None = None, *, with_memory: bool = False
) -> (
    Callable[..., tuple[T, AgentContextTrace]]
    | Callable[[Callable[..., T]], Callable[..., tuple[T, AgentContextTrace]]]
):
    """Decorator that creates per-call AgentContextTrace.

    Supports both:
    - `@agentic_trace_root`
    - `@agentic_trace_root(with_memory=True)`
    """
    if func is None:
        return lambda inner: _decorate_agentic_root(inner, with_memory=with_memory)
    return _decorate_agentic_root(func, with_memory=with_memory)
