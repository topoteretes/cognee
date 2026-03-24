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


def _start_agent_context_trace(origin_function: str) -> tuple[AgentContextTrace, object]:
    context_trace = AgentContextTrace(origin_function=origin_function)
    token = _current_agent_context_trace.set(context_trace)
    return context_trace, token


def agentic_trace_root(func: Callable[..., T]) -> Callable[..., tuple[T, AgentContextTrace]]:
    """Decorator that creates a per-call AgentContextTrace for sync and async entrypoints."""

    if inspect.iscoroutinefunction(func):
        async_func = cast(Callable[..., Awaitable[T]], func)

        @wraps(async_func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> tuple[T, AgentContextTrace]:
            context_trace, token = _start_agent_context_trace(async_func.__name__)
            try:
                result = await async_func(*args, **kwargs)
                return result, context_trace
            finally:
                _current_agent_context_trace.reset(token)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> tuple[T, AgentContextTrace]:
        context_trace, token = _start_agent_context_trace(func.__name__)
        try:
            result = func(*args, **kwargs)
            return result, context_trace
        finally:
            _current_agent_context_trace.reset(token)

    return sync_wrapper
