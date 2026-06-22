from __future__ import annotations

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
from cognee.modules.agent_memory.sanitization import (
    MAX_SERIALIZED_VALUE_LENGTH,
    sanitize_value,
    truncate_text,
)

logger = get_logger("agent_memory")

MAX_MEMORY_CONTEXT_LENGTH = 4000


@dataclass(slots=True)
class AgentMemoryConfig:
    """Validated decorator configuration used for one wrapped agent invocation."""

    with_memory: bool
    with_session_memory: bool
    save_session_traces: bool
    memory_query_fixed: Optional[str]
    memory_query_from_method: Optional[str]
    memory_system_prompt: Optional[str]
    memory_top_k: int
    memory_only_context: bool
    session_memory_last_n: int
    session_id: Optional[str]
    user: Optional[User]
    dataset_name: Optional[str]
    session_trace_summary: bool
    persist_session_trace_after: Optional[int]
    persist_session_trace_raw_content: bool
    persist_session_trace_node_set_name: Optional[str]


@dataclass(slots=True)
class AgentScope:
    """Authorized dataset scope resolved for Cognee-backed memory retrieval."""

    user: User
    dataset_name: str
    dataset_id: UUID


@dataclass(slots=True)
class AgentMemoryContext:
    """Per-call execution state shared across retrieval, wrapped call, and trace persistence."""

    origin_function: str
    config: AgentMemoryConfig
    method_params: dict[str, Any]
    user: Optional[User] = None
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
    with_session_memory: bool,
    save_session_traces: bool,
    memory_query_fixed: Optional[str],
    memory_query_from_method: Optional[str],
    memory_system_prompt: Optional[str],
    memory_top_k: int,
    memory_only_context: bool,
    session_memory_last_n: int,
    session_id: Optional[str],
    user: Optional[User],
    dataset_name: Optional[str],
    session_trace_summary: bool,
    persist_session_trace_after: Optional[int],
    persist_session_trace_raw_content: bool,
    persist_session_trace_node_set_name: Optional[str],
) -> AgentMemoryConfig:
    """Validate and normalize the public decorator configuration."""
    from cognee.infrastructure.databases.cache.config import get_cache_config

    if not isinstance(with_memory, bool):
        raise CogneeValidationError("with_memory must be a boolean.", log=False)
    if not isinstance(with_session_memory, bool):
        raise CogneeValidationError("with_session_memory must be a boolean.", log=False)
    if not isinstance(save_session_traces, bool):
        raise CogneeValidationError("save_session_traces must be a boolean.", log=False)
    if not isinstance(memory_only_context, bool):
        raise CogneeValidationError("memory_only_context must be a boolean.", log=False)
    if not isinstance(session_trace_summary, bool):
        raise CogneeValidationError("session_trace_summary must be a boolean.", log=False)
    if not isinstance(persist_session_trace_raw_content, bool):
        raise CogneeValidationError(
            "persist_session_trace_raw_content must be a boolean.",
            log=False,
        )
    if persist_session_trace_node_set_name is not None and not isinstance(
        persist_session_trace_node_set_name, str
    ):
        raise CogneeValidationError(
            "persist_session_trace_node_set_name must be a string when provided.",
            log=False,
        )
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
    if (
        persist_session_trace_node_set_name is not None
        and not persist_session_trace_node_set_name.strip()
    ):
        raise CogneeValidationError(
            "persist_session_trace_node_set_name must not be blank when provided.",
            log=False,
        )
    if memory_query_fixed is not None and memory_query_from_method is not None:
        raise CogneeValidationError(
            "Only one of memory_query_fixed or memory_query_from_method can be provided to cognee.agent_memory.",
            log=False,
        )
    if not isinstance(session_memory_last_n, int) or session_memory_last_n < 1:
        raise CogneeValidationError(
            "session_memory_last_n must be a positive integer.",
            log=False,
        )
    if persist_session_trace_after is not None and (
        not isinstance(persist_session_trace_after, int) or persist_session_trace_after < 1
    ):
        raise CogneeValidationError(
            "persist_session_trace_after must be a positive integer when provided.",
            log=False,
        )
    if persist_session_trace_after is not None and not save_session_traces:
        raise CogneeValidationError(
            "persist_session_trace_after requires save_session_traces=True.",
            log=False,
        )
    cache_config = get_cache_config()
    if not cache_config.caching and (
        with_session_memory or save_session_traces or persist_session_trace_after is not None
    ):
        raise CogneeValidationError(
            (
                "Caching must be enabled to use with_session_memory, save_session_traces, "
                "or persist_session_trace_after with cognee.agent_memory."
            ),
            log=False,
        )
    if persist_session_trace_after is not None and (
        not isinstance(persist_session_trace_after, int) or persist_session_trace_after < 1
    ):
        raise CogneeValidationError(
            "persist_session_trace_after must be a positive integer when provided.",
            log=False,
        )
    if persist_session_trace_after is not None and not save_session_traces:
        raise CogneeValidationError(
            "persist_session_trace_after requires save_session_traces=True.",
            log=False,
        )
    cache_config = get_cache_config()
    if not cache_config.caching and (
        with_session_memory or save_session_traces or persist_session_trace_after is not None
    ):
        raise CogneeValidationError(
            (
                "Caching must be enabled to use with_session_memory, save_session_traces, "
                "or persist_session_trace_after with cognee.agent_memory."
            ),
            log=False,
        )
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise CogneeValidationError(
            "session_id must be a non-empty string when provided.",
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
        with_session_memory=with_session_memory,
        save_session_traces=save_session_traces,
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
        memory_only_context=memory_only_context,
        session_memory_last_n=session_memory_last_n,
        session_id=session_id.strip() if isinstance(session_id, str) else None,
        user=user,
        dataset_name=dataset_name.strip() if isinstance(dataset_name, str) else None,
        session_trace_summary=session_trace_summary,
        persist_session_trace_after=persist_session_trace_after,
        persist_session_trace_raw_content=persist_session_trace_raw_content,
        persist_session_trace_node_set_name=(
            persist_session_trace_node_set_name.strip()
            if isinstance(persist_session_trace_node_set_name, str)
            else None
        ),
    )


async def resolve_agent_user(config: AgentMemoryConfig) -> User:
    """Resolve the effective user for agent-memory search/session operations."""
    return config.user or await get_default_user()


async def resolve_agent_dataset_scope(config: AgentMemoryConfig, resolved_user: User) -> AgentScope:
    """Resolve the dataset scope for Cognee search using a user with read and write access."""
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
        if key in {"user", "dataset_name", "session_id"}:
            continue
        if not isinstance(value, str):
            continue
        normalized_value = normalize_optional_text(value)
        if normalized_value:
            return normalized_value

    return None


async def retrieve_memory_context(context: AgentMemoryContext) -> str:
    """Fetch memory text for the current agent execution across enabled memory sources."""
    memory_parts: list[str] = []

    session_memory = await retrieve_session_memory_context(context)
    if session_memory:
        memory_parts.append(f"Recent Session Memory:\n{session_memory}")

    cognee_memory = await retrieve_cognee_memory_context(context)
    if cognee_memory:
        memory_parts.append(f"Relevant Cognee Memory:\n{cognee_memory}")

    if not memory_parts:
        return ""

    return truncate_text("\n\n".join(memory_parts), MAX_MEMORY_CONTEXT_LENGTH)


async def retrieve_cognee_memory_context(context: AgentMemoryContext) -> str:
    """Fetch dataset-backed Cognee search memory when enabled."""
    if not context.config.with_memory or context.scope is None:
        context.memory_query = ""
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
                only_context=context.config.memory_only_context,
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


async def retrieve_session_memory_context(context: AgentMemoryContext) -> str:
    """Fetch recent trace feedback from the session-backed trace store when enabled."""
    if not context.config.with_session_memory or context.user is None:
        return ""

    from cognee.infrastructure.session.get_session_manager import get_session_manager

    session_manager = get_session_manager()
    try:
        feedback_values = await session_manager.get_agent_trace_feedback(
            user_id=str(context.user.id),
            session_id=context.config.session_id,
            last_n=context.config.session_memory_last_n,
        )
    except Exception as error:
        logger.warning(
            "Session agent memory retrieval failed for %s: %s",
            context.origin_function,
            error,
            exc_info=False,
        )
        return ""

    normalized_feedback = [
        normalized
        for value in feedback_values
        if (normalized := normalize_optional_text(value)) is not None
    ]
    if not normalized_feedback:
        return ""

    return "\n".join(normalized_feedback)


async def persist_trace(context: AgentMemoryContext) -> None:
    """Persist one agent trace step into session-backed storage."""
    if not context.config.save_session_traces or context.user is None:
        return

    from cognee.infrastructure.session.get_session_manager import get_session_manager

    session_manager = get_session_manager()
    user_id = str(context.user.id)
    try:
        await session_manager.add_agent_trace_step(
            user_id=user_id,
            session_id=context.config.session_id,
            origin_function=context.origin_function,
            status=context.status,
            generate_feedback_with_llm=context.config.session_trace_summary,
            memory_query=context.memory_query,
            memory_context=context.memory_context,
            method_params=context.method_params,
            method_return_value=sanitize_value(context.method_return_value),
            error_message=truncate_text(context.error_message, MAX_SERIALIZED_VALUE_LENGTH),
        )
    except Exception as error:
        logger.warning(
            "Agent trace persistence failed for %s: %s",
            context.origin_function,
            error,
            exc_info=False,
        )
        return

    if context.config.persist_session_trace_after is None:
        return

    resolved_session_id = context.config.session_id or session_manager.default_session_id

    try:
        trace_count = await session_manager.get_agent_trace_count(
            user_id=user_id,
            session_id=resolved_session_id,
        )
        if trace_count == 0 or trace_count % context.config.persist_session_trace_after != 0:
            return

        from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
            persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
        )

        persist_kwargs = {
            "user": context.user,
            "session_ids": [resolved_session_id],
            "dataset": context.config.dataset_name or "main_dataset",
            "raw_trace_content": context.config.persist_session_trace_raw_content,
            "last_n_steps": context.config.persist_session_trace_after,
            "run_in_background": False,
        }
        if context.config.persist_session_trace_node_set_name is not None:
            persist_kwargs["node_set_name"] = context.config.persist_session_trace_node_set_name

        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            **persist_kwargs,
        )
    except Exception as error:
        logger.warning(
            "Agent trace memify persistence failed for %s: %s",
            context.origin_function,
            error,
            exc_info=False,
        )


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
