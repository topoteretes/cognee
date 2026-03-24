from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Any, Callable, Optional

from .prompt_trace_context import AgentContextTrace

_agent_context_trace_var: contextvars.ContextVar[Optional[AgentContextTrace]] = (
    contextvars.ContextVar("agent_context_trace", default=None)
)


def get_current_agent_context_trace() -> Optional[AgentContextTrace]:
    return _agent_context_trace_var.get()


def agentic_trace_root(*, with_memory: bool = False, task_query: str = "") -> Callable[
    [Callable[..., Any]], Callable[..., Any]
]:
    """Decorate an async root coroutine so AgentContextTrace is available via ContextVar."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"agentic_trace_root requires an async function; got {fn!r}")

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound_args = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            trace = AgentContextTrace(
                origin_function=fn.__qualname__,
                with_memory=with_memory,
                task_query=task_query,
            )
            # TODO: Later we can add a decorator parameter to control which method params are persisted, it is safer like that.
            trace.method_params = dict(bound_args.arguments)
            token = _agent_context_trace_var.set(trace)
            try:
                if with_memory:
                    await trace.get_memory_context(trace.task_query or "")
                result = await fn(*args, **kwargs)
                trace.method_return_value = result
                return result
            finally:
                _agent_context_trace_var.reset(token)

        return wrapper

    return decorator
