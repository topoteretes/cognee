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


def _start_agent_context_trace(
    origin_function: str, with_memory: bool, task_query: str
) -> tuple[AgentContextTrace, object]:
    context_trace = AgentContextTrace(
        origin_function=origin_function,
        with_memory=with_memory,
        task_query=task_query,
    )
    token = _current_agent_context_trace.set(context_trace)
    return context_trace, token


def agentic_trace_root(
    *, with_memory: bool = False, task_query: str = ""
) -> Callable[[Callable[..., T]], Callable[..., tuple[T, AgentContextTrace]]]:
    """Simple async-only decorator factory."""

    def decorator(func: Callable[..., T]) -> Callable[..., tuple[T, AgentContextTrace]]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("agentic_trace_root supports only async functions.")

        async_func = cast(Callable[..., Awaitable[T]], func)

        @wraps(async_func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> tuple[T, AgentContextTrace]:
            context_trace, token = _start_agent_context_trace(
                async_func.__name__, with_memory=with_memory, task_query=task_query
            )
            try:
                if context_trace.with_memory and context_trace.task_query:
                    await context_trace.get_memory_context(context_trace.task_query)
                result = await async_func(*args, **kwargs)
                return result, context_trace
            finally:
                _current_agent_context_trace.reset(token)

        return async_wrapper

    return decorator
