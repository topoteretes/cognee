from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from cognee.exceptions import CogneeValidationError
from cognee.modules.users.models import User

from .runtime import (
    AgentMemoryContext,
    build_method_params,
    persist_trace,
    reset_current_agent_memory_context,
    resolve_agent_scope,
    retrieve_memory_context,
    set_current_agent_memory_context,
    validate_agent_memory_config,
)


def agent_memory(
    *,
    with_memory: bool = True,
    save_traces: bool = False,
    memory_query_fixed: Optional[str] = None,
    memory_query_from_method: Optional[str] = None,
    memory_system_prompt: Optional[str] = None,
    memory_top_k: int = 5,
    user: Optional[User] = None,
    dataset_name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an async agent entrypoint with optional Cognee memory and trace persistence."""
    config = validate_agent_memory_config(
        with_memory=with_memory,
        save_traces=save_traces,
        memory_query_fixed=memory_query_fixed,
        memory_query_from_method=memory_query_from_method,
        memory_system_prompt=memory_system_prompt,
        memory_top_k=memory_top_k,
        user=user,
        dataset_name=dataset_name,
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
            scope = None
            if config.with_memory or config.save_traces:
                scope = await resolve_agent_scope(config)
            context = AgentMemoryContext(
                origin_function=fn.__qualname__,
                config=config,
                method_params=build_method_params(fn, args, kwargs),
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
                await persist_trace(context)
                reset_current_agent_memory_context(token)

        return wrapper

    return decorator
