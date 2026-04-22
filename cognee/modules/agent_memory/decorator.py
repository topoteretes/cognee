from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from cognee.exceptions import CogneeValidationError
from cognee.modules.users.models import User

from cognee.modules.agent_memory.runtime import (
    AgentMemoryContext,
    build_method_params,
    persist_trace,
    reset_current_agent_memory_context,
    resolve_agent_dataset_scope,
    resolve_agent_user,
    retrieve_memory_context,
    set_current_agent_memory_context,
    validate_agent_memory_config,
)


def agent_memory(
    *,
    with_memory: bool = True,
    with_session_memory: bool = False,
    save_session_traces: bool = False,
    memory_query_fixed: Optional[str] = None,
    memory_query_from_method: Optional[str] = None,
    memory_system_prompt: Optional[str] = None,
    memory_top_k: int = 5,
    memory_only_context: bool = False,
    session_memory_last_n: int = 5,
    session_id: Optional[str] = None,
    user: Optional[User] = None,
    dataset_name: Optional[str] = None,
    session_trace_summary: bool = True,
    persist_session_trace_after: Optional[int] = None,
    persist_session_trace_raw_content: bool = False,
    persist_session_trace_node_set_name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate an async agent entrypoint with optional Cognee memory and trace persistence.

    We strongly recommend using a dedicated session per decorated method. Reusing the same
    ``session_id`` across different decorated entrypoints can mix unrelated trace history and
    make session-memory retrieval or periodic trace memify harder to reason about, especially
    when those decorators do not share the same trace-persistence settings.
    """
    config = validate_agent_memory_config(
        with_memory=with_memory,
        with_session_memory=with_session_memory,
        save_session_traces=save_session_traces,
        memory_query_fixed=memory_query_fixed,
        memory_query_from_method=memory_query_from_method,
        memory_system_prompt=memory_system_prompt,
        memory_top_k=memory_top_k,
        memory_only_context=memory_only_context,
        session_memory_last_n=session_memory_last_n,
        session_id=session_id,
        user=user,
        dataset_name=dataset_name,
        session_trace_summary=session_trace_summary,
        persist_session_trace_after=persist_session_trace_after,
        persist_session_trace_raw_content=persist_session_trace_raw_content,
        persist_session_trace_node_set_name=persist_session_trace_node_set_name,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.iscoroutinefunction(fn):
            raise CogneeValidationError(
                f"cognee.agent_memory requires an async function; got {fn!r}",
                log=False,
            )
        if config.memory_query_from_method:
            fn_params = inspect.signature(fn).parameters
            if config.memory_query_from_method not in fn_params:
                raise CogneeValidationError(
                    (
                        f"memory_query_from_method={config.memory_query_from_method!r} "
                        f"does not match a parameter on {fn.__qualname__}."
                    ),
                    log=False,
                )

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_user = None
            scope = None
            if config.with_memory or config.with_session_memory or config.save_session_traces:
                resolved_user = await resolve_agent_user(config)
                if config.with_memory:
                    scope = await resolve_agent_dataset_scope(config, resolved_user)
            context = AgentMemoryContext(
                origin_function=fn.__qualname__,
                config=config,
                method_params=build_method_params(fn, args, kwargs),
                user=resolved_user,
                scope=scope,
            )
            token = set_current_agent_memory_context(context)

            try:
                context.memory_context = await retrieve_memory_context(context)
                result = await fn(*args, **kwargs)
                context.method_return_value = result
                context.status = "success"
                return result
            except Exception as error:
                context.status = "error"
                context.error_message = str(error)
                raise
            finally:
                reset_current_agent_memory_context(token)
                await persist_trace(context)

        return wrapper

    return decorator
