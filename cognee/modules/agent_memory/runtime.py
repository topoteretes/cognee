from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from cognee.exceptions import CogneeValidationError
from cognee.modules.observability import new_span
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import get_all_user_permission_datasets
from cognee.shared.logging_utils import get_logger

from .models import AgentTrace
from cognee.context_global_variables import set_database_global_context_variables

logger = get_logger("agent_memory")

MAX_SERIALIZED_VALUE_LENGTH = 1000
MAX_MEMORY_CONTEXT_LENGTH = 4000
MAX_TRACE_TEXT_LENGTH = 4000


@dataclass(slots=True)
class AgentMemoryConfig:
    with_memory: bool
    save_traces: bool
    memory_query_fixed: Optional[str]
    memory_query_from_method: Optional[str]
    memory_system_prompt: Optional[str]
    memory_top_k: int
    user: Optional[User]
    dataset_name: Optional[str]


@dataclass(slots=True)
class AgentScope:
    user: User
    dataset_name: str
    dataset_id: UUID
    dataset_owner_id: Optional[UUID]


@dataclass(slots=True)
class AgentMemoryContext:
    origin_function: str
    config: AgentMemoryConfig
    method_params: dict[str, Any]
    scope: Optional[AgentScope] = None
    memory_query: str = ""
    memory_context: str = ""
    method_return_value: Any = None
    status: str = "running"
    error_message: str = ""


_agent_memory_context_var: contextvars.ContextVar[Optional[AgentMemoryContext]] = (
    contextvars.ContextVar("agent_memory_context", default=None)
)


def get_current_agent_memory_context() -> Optional[AgentMemoryContext]:
    """Return the active agent-memory execution context for the current async task."""
    return _agent_memory_context_var.get()


def set_current_agent_memory_context(
    context: AgentMemoryContext,
) -> contextvars.Token[Optional[AgentMemoryContext]]:
    """Store the active agent-memory context and return a reset token."""
    return _agent_memory_context_var.set(context)


def reset_current_agent_memory_context(
    token: contextvars.Token[Optional[AgentMemoryContext]],
) -> None:
    """Restore the previously active agent-memory context."""
    _agent_memory_context_var.reset(token)


def validate_agent_memory_config(
    *,
    with_memory: bool,
    save_traces: bool,
    memory_query_fixed: Optional[str],
    memory_query_from_method: Optional[str],
    memory_system_prompt: Optional[str],
    memory_top_k: int,
    user: Optional[User],
    dataset_name: Optional[str],
) -> AgentMemoryConfig:
    """Validate and normalize the public decorator configuration."""
    if not isinstance(with_memory, bool):
        raise CogneeValidationError("with_memory must be a boolean.", log=False)
    if not isinstance(save_traces, bool):
        raise CogneeValidationError("save_traces must be a boolean.", log=False)
    if memory_query_fixed is not None and not isinstance(memory_query_fixed, str):
        raise CogneeValidationError("memory_query_fixed must be a string when provided.", log=False)
    if memory_query_from_method is not None and not isinstance(memory_query_from_method, str):
        raise CogneeValidationError(
            "memory_query_from_method must be a string when provided.",
            log=False,
        )
    if memory_system_prompt is not None and not isinstance(memory_system_prompt, str):
        raise CogneeValidationError(
            "memory_system_prompt must be a string when provided.",
            log=False,
        )
    if memory_query_fixed is not None and not memory_query_fixed.strip():
        raise CogneeValidationError(
            "memory_query_fixed must not be blank when provided.",
            log=False,
        )
    if memory_query_from_method is not None and not memory_query_from_method.strip():
        raise CogneeValidationError(
            "memory_query_from_method must not be blank when provided.",
            log=False,
        )
    if memory_system_prompt is not None and not memory_system_prompt.strip():
        raise CogneeValidationError(
            "memory_system_prompt must not be blank when provided.",
            log=False,
        )
    if memory_query_fixed is not None and memory_query_from_method is not None:
        raise CogneeValidationError(
            "Only one of memory_query_fixed or memory_query_from_method can be provided to cognee.agent_memory.",
            log=False,
        )
    if user is not None and not hasattr(user, "id"):
        raise CogneeValidationError("user must have an id attribute.", log=False)
    if dataset_name is not None and (not isinstance(dataset_name, str) or not dataset_name.strip()):
        raise CogneeValidationError(
            "dataset_name must be a non-empty string when provided.",
            log=False,
        )

    return AgentMemoryConfig(
        with_memory=with_memory,
        save_traces=save_traces,
        memory_query_fixed=(
            memory_query_fixed.strip() if isinstance(memory_query_fixed, str) else None
        ),
        memory_query_from_method=(
            memory_query_from_method.strip() if isinstance(memory_query_from_method, str) else None
        ),
        memory_system_prompt=(
            memory_system_prompt.strip() if isinstance(memory_system_prompt, str) else None
        ),
        memory_top_k=memory_top_k,
        user=user,
        dataset_name=dataset_name.strip() if isinstance(dataset_name, str) else None,
    )


async def resolve_agent_scope(config: AgentMemoryConfig) -> AgentScope:
    """Resolve the dataset scope for a user who must have both read and write access."""
    resolved_user = config.user or await get_default_user()
    requested_dataset_name = config.dataset_name or "main_dataset"

    readable_datasets = await get_all_user_permission_datasets(resolved_user, "read")
    writable_datasets = await get_all_user_permission_datasets(resolved_user, "write")

    readable_by_id = {dataset.id: dataset for dataset in readable_datasets}
    writable_ids = {dataset.id: dataset for dataset in writable_datasets}
    matching_datasets = [
        dataset
        for dataset in readable_by_id.values()
        if dataset.id in writable_ids and dataset.name == requested_dataset_name
    ]

    if len(matching_datasets) > 1:
        raise CogneeValidationError(
            (
                f"Multiple datasets named {requested_dataset_name!r} grant both read and write "
                f"permissions to user {resolved_user.id}. Please use a unique dataset name."
            ),
            log=False,
        )
    if not matching_datasets:
        raise CogneeValidationError(
            (
                f"User {resolved_user.id} must have both read and write permissions for dataset "
                f"{requested_dataset_name!r} to use cognee.agent_memory."
            ),
            log=False,
        )

    authorized_dataset = matching_datasets[0]

    return AgentScope(
        user=resolved_user,
        dataset_name=authorized_dataset.name,
        dataset_id=authorized_dataset.id,
        dataset_owner_id=authorized_dataset.owner_id,
    )


def build_method_params(func, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind wrapped call arguments to parameter names and sanitize them for storage."""
    bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    return {key: sanitize_value(value) for key, value in bound_args.arguments.items()}


def normalize_optional_text(value: Any, limit: int = MAX_SERIALIZED_VALUE_LENGTH) -> Optional[str]:
    """Convert a value into a bounded non-empty string, or return None when unusable."""
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(sanitize_value(value))

    stripped = value.strip()
    if not stripped:
        return None

    return stripped[:limit]


def get_query_text_from_method_param(
    memory_query_from_method: Optional[str],
    method_params: dict[str, Any],
) -> Optional[str]:
    """Extract a bounded retrieval query from a configured wrapped-method parameter."""
    if not memory_query_from_method or memory_query_from_method not in method_params:
        return None

    return normalize_optional_text(method_params[memory_query_from_method])


def derive_query_text(
    memory_query_fixed: Optional[str],
    memory_query_from_method: Optional[str],
    method_params: dict[str, Any],
) -> Optional[str]:
    """Resolve the retrieval query from dynamic, fixed, or fallback method inputs."""
    query_from_method = get_query_text_from_method_param(memory_query_from_method, method_params)
    if query_from_method:
        return query_from_method

    if memory_query_fixed:
        return memory_query_fixed

    for key, value in method_params.items():
        if key in {"user", "dataset_name"}:
            continue
        if not isinstance(value, str):
            continue
        normalized_value = normalize_optional_text(value)
        if normalized_value:
            return normalized_value

    return None


async def retrieve_memory_context(context: AgentMemoryContext) -> str:
    """Fetch memory text for the current agent execution using the resolved dataset scope."""
    if not context.config.with_memory:
        return ""
    if context.scope is None:
        return ""

    query_text = derive_query_text(
        context.config.memory_query_fixed,
        context.config.memory_query_from_method,
        context.method_params,
    )
    context.memory_query = query_text or ""
    if not query_text:
        logger.info("Skipping agent memory retrieval because no usable query could be derived.")
        return ""

    with new_span("cognee.agent_memory.retrieve") as span:
        span.set_attribute("cognee.agent_memory.query_length", len(query_text))
        try:
            from cognee.api.v1.search import SearchType, search

            results = await search(
                query_text=query_text,
                query_type=SearchType.GRAPH_SUMMARY_COMPLETION,
                user=context.scope.user,
                dataset_ids=[context.scope.dataset_id],
                system_prompt=context.config.memory_system_prompt,
                top_k=context.config.memory_top_k,
            )
        except Exception as error:
            logger.warning(
                "Agent memory retrieval failed for %s: %s",
                context.origin_function,
                error,
                exc_info=False,
            )
            span.set_attribute("cognee.agent_memory.retrieval_failed", True)
            return ""

        memory_context = truncate_text(normalize_search_results(results), MAX_MEMORY_CONTEXT_LENGTH)
        span.set_attribute("cognee.agent_memory.context_length", len(memory_context))
        return memory_context


async def persist_trace(context: AgentMemoryContext) -> None:
    """Persist a bounded agent trace after execution in an isolated async task context."""
    if not context.config.save_traces:
        return
    if context.scope is None:
        return

    trace = build_agent_trace(context)

    async def _persist_trace_in_task() -> None:
        from cognee.tasks.storage import add_data_points

        await set_database_global_context_variables(
            context.scope.dataset_id,
            context.scope.dataset_owner_id,
        )
        await add_data_points([trace])

    try:
        await asyncio.create_task(_persist_trace_in_task())
    except Exception as error:
        logger.warning(
            "Agent trace persistence failed for %s: %s",
            context.origin_function,
            error,
            exc_info=False,
        )


def build_trace_text(context: AgentMemoryContext) -> str:
    """Build the lean searchable text field stored on the persisted trace."""
    output_text = serialize_trace_output(context.method_return_value)
    if output_text:
        return truncate_text(output_text, MAX_TRACE_TEXT_LENGTH)

    return truncate_text(context.error_message, MAX_TRACE_TEXT_LENGTH)


def build_agent_trace(context: AgentMemoryContext) -> AgentTrace:
    """Create the structured trace payload persisted for one agent execution."""
    # TODO: Redact or further constrain method_params and method_return_value before
    # persisting them, since truncation alone does not prevent secrets or PII retention.
    return AgentTrace(
        origin_function=context.origin_function,
        with_memory=context.config.with_memory,
        memory_query=context.memory_query,
        method_params=context.method_params,
        method_return_value=sanitize_value(context.method_return_value),
        memory_context=context.memory_context,
        status=context.status,
        error_message=truncate_text(context.error_message, MAX_SERIALIZED_VALUE_LENGTH),
        text=build_trace_text(context),
    )


def serialize_trace_output(value: Any) -> str:
    """Serialize a sanitized return value into a trace-friendly string."""
    sanitized_output = sanitize_value(value)
    if isinstance(sanitized_output, (dict, list)):
        return json.dumps(sanitized_output, default=str, ensure_ascii=False)
    if sanitized_output is None:
        return ""
    return str(sanitized_output)


def normalize_search_results(results: Any) -> str:
    """Flatten heterogeneous search outputs into a single text blob."""
    if results is None:
        return ""
    if isinstance(results, str):
        return results
    if isinstance(results, list):
        normalized_items = [normalize_search_results(item) for item in results]
        return "\n".join(item for item in normalized_items if item).strip()
    if isinstance(results, dict):
        if "search_result" in results:
            return normalize_search_results(results["search_result"])
        return json.dumps(sanitize_value(results), ensure_ascii=False)
    if hasattr(results, "search_result"):
        return normalize_search_results(results.search_result)
    return str(results)


def sanitize_value(value: Any) -> Any:
    """Convert arbitrary runtime values into bounded, persistence-safe structures."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return truncate_text(value, MAX_SERIALIZED_VALUE_LENGTH)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value[:20]]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value[:20]]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:20]:
            sanitized[str(key)] = sanitize_value(item)
        return sanitized
    if hasattr(value, "id") and hasattr(value, "__class__"):
        return {
            "type": value.__class__.__name__,
            "id": str(getattr(value, "id", "")),
        }
    return truncate_text(str(value), MAX_SERIALIZED_VALUE_LENGTH)


def truncate_text(value: str, limit: int) -> str:
    """Truncate text to a fixed limit while preserving an ellipsis suffix."""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
