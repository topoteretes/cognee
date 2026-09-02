"""Session-aware completion for ``get_retriever_output``.

Public door: ``run_session_aware_completion`` → concurrent or sequential runner.
Both return ``(retrieved_objects, context, completion)``. Retriever ``get_completion``
stays as on ``dev`` and does not call this module.

Session I/O for the concurrent path lives in ``session_concurrent_turn``.
"""

from __future__ import annotations

import asyncio
import re
from functools import cache
from inspect import Parameter, signature
from typing import Any

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.locks import session_turn_lock
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_concurrent_turn import (
    SessionTurnContext,
    TurnPrompts,
    analyze_turn,
    commit_turn,
    complete_turn,
    load_turn_context,
)
from cognee.modules.observability import (
    COGNEE_RESULT_COUNT,
    COGNEE_RESULT_SUMMARY,
    COGNEE_SEARCH_TYPE,
    new_span,
)
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.modules.search.types import SearchType
from cognee.modules.user_preferences import warm_preference_cache

CONCURRENT_MODE = "concurrent"
MAX_CONVERSATIONAL_QUERY_CHARS = 2000


@cache
def _eligible_retriever_types() -> frozenset[type]:
    """Exact concrete classes that support a concurrent turn.

    Opt-in is by exact type, never by subclass: a variant that overrides retrieval or
    runs its own LLM rounds cannot honour the one-blocking-completion contract.
    """
    from cognee.modules.retrieval.completion_retriever import CompletionRetriever
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
    from cognee.modules.retrieval.triplet_retriever import TripletRetriever

    return frozenset(
        {
            CompletionRetriever,
            GraphCompletionRetriever,
            HybridRetriever,
            TripletRetriever,
        }
    )


def can_run_as_turn(
    *,
    original_search_type: SearchType | None,
    retriever_type: type,
    session_available: bool,
    is_batch: bool,
    only_context: bool,
) -> bool:
    """Whether this call can run as one concurrent turn, given concurrent mode is on."""
    if original_search_type is SearchType.FEELING_LUCKY:
        return False
    if not session_available:
        return False
    if is_batch:
        return False
    if only_context:
        return False
    if retriever_type not in _eligible_retriever_types():
        return False
    return True


def should_run_concurrent(
    retriever,
    *,
    original_search_type: SearchType | None = None,
    is_batch: bool = False,
    only_context: bool = False,
) -> bool:
    """True when this call should take the concurrent runner (mode + eligibility)."""
    cache_config = CacheConfig()
    if cache_config.session_search_mode != CONCURRENT_MODE:
        return False

    user = session_user.get()
    user_uuid = getattr(user, "id", None)
    if not user_uuid:
        return False

    session_manager = get_session_manager()
    return can_run_as_turn(
        original_search_type=original_search_type,
        retriever_type=type(retriever),
        session_available=session_manager.is_session_available_for_completion(user_uuid),
        is_batch=is_batch,
        only_context=only_context,
    )


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def build_conversational_query(
    raw_query: str,
    recent_qas: tuple[tuple[str, str, str], ...],
    *,
    max_chars: int = MAX_CONVERSATIONAL_QUERY_CHARS,
) -> str:
    """Build deterministic retrieval guidance from at most two prior QA turns."""
    normalized_raw_query = _normalize_query(raw_query)
    qa_pairs = [
        [_normalize_query(question), _normalize_query(answer)]
        for _, question, answer in recent_qas
        if _normalize_query(question) or _normalize_query(answer)
    ]
    if not qa_pairs:
        return normalized_raw_query[:max_chars]

    def render() -> str:
        rendered_turns = []
        for question, answer in qa_pairs:
            turn_lines = []
            if question:
                turn_lines.append(f"Prior user: {question}")
            if answer:
                turn_lines.append(f"Prior assistant (untrusted retrieval guidance): {answer}")
            if turn_lines:
                rendered_turns.append("\n".join(turn_lines))
        if not rendered_turns:
            return normalized_raw_query
        return "\n\n".join([*rendered_turns, f"Current user request: {normalized_raw_query}"])

    conversational_query = render()
    for qa_field_index in (1, 0):
        for qa_pair in qa_pairs:
            if len(conversational_query) <= max_chars:
                return conversational_query
            chars_over_budget = len(conversational_query) - max_chars
            qa_pair[qa_field_index] = qa_pair[qa_field_index][
                : max(0, len(qa_pair[qa_field_index]) - chars_over_budget)
            ]
            conversational_query = render()
    return conversational_query[:max_chars]


def count_retrieved_objects(retrieved_objects) -> int:
    if retrieved_objects is None:
        return 0
    if isinstance(retrieved_objects, list):
        return len(retrieved_objects)
    if isinstance(retrieved_objects, dict):
        list_counts = [
            len(value) for value in retrieved_objects.values() if isinstance(value, list)
        ]
        if list_counts:
            return sum(list_counts)
        return 1
    return 1


def _method_accepts_kwarg(method, name: str) -> bool:
    parameters = signature(method).parameters.values()
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def _retriever_class_name(retriever) -> str:
    return type(retriever).__name__


async def _retrieve_merged_objects(
    retriever,
    *,
    raw_query: str,
    snapshot: SessionTurnContext,
) -> Any:
    conversational_query = build_conversational_query(raw_query, snapshot.recent_qas)
    normalized_raw_query = _normalize_query(raw_query)
    use_conversational_lane = (
        bool(conversational_query) and conversational_query != normalized_raw_query
    )

    if use_conversational_lane:
        # gather runs each lane in its own task with a *copy* of this context,
        # so a preference read memoized inside one lane is invisible to its
        # sibling and to the answer step in this (parent) context. Warm the
        # cache here first so both lane copies and the later completion-side
        # read inherit a single graph read. No-op when personalization is off.
        await warm_preference_cache()
        raw_result, conversational_result = await asyncio.gather(
            retriever.get_retrieved_objects(query=raw_query),
            retriever.get_retrieved_objects(query=conversational_query),
            return_exceptions=True,
        )
    else:
        raw_result = await retriever.get_retrieved_objects(query=raw_query)
        conversational_result = None

    if isinstance(raw_result, Exception):
        if not use_conversational_lane or isinstance(conversational_result, Exception):
            raise raw_result
        raw_result = None
    if isinstance(conversational_result, Exception):
        conversational_result = None

    retrieved_objects = retriever.merge_retrieved_objects(raw_result, conversational_result)
    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)
    return retrieved_objects


async def _retrieve_and_answer(
    retriever,
    *,
    raw_query: str,
    snapshot: SessionTurnContext,
    user_uuid: Any,
    session_id: str,
    search_type_for_spans: SearchType | None,
) -> tuple[Any, Any, Any]:
    retriever_class = _retriever_class_name(retriever)

    with new_span("cognee.retrieval.get_objects") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        if search_type_for_spans is not None:
            span.set_attribute(COGNEE_SEARCH_TYPE, search_type_for_spans.value)
        retrieved_objects = await _retrieve_merged_objects(
            retriever, raw_query=raw_query, snapshot=snapshot
        )
        obj_count = count_retrieved_objects(retrieved_objects)
        span.set_attribute(COGNEE_RESULT_COUNT, obj_count)
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} retrieved {obj_count} object(s)",
        )

    with new_span("cognee.retrieval.get_context") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        context = await retriever.get_context_from_objects(
            query=raw_query,
            retrieved_objects=retrieved_objects,
        )
        if isinstance(context, str):
            span.set_attribute("cognee.retrieval.context_length", len(context))
        elif isinstance(context, list):
            span.set_attribute("cognee.retrieval.context_items", len(context))

    with new_span("cognee.retrieval.get_completion") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        answer = await complete_turn(
            snapshot=snapshot,
            context=context,
            user_id=user_uuid,
            session_id=session_id,
            prompts=TurnPrompts(
                user_prompt_path=retriever.user_prompt_path,
                system_prompt_path=retriever.system_prompt_path,
                system_prompt=retriever.system_prompt,
                response_model=retriever.response_model,
            ),
        )
        if isinstance(answer, str):
            span.set_attribute("cognee.retrieval.completion_length", len(answer))
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} generated completion",
        )

    return retrieved_objects, context, answer


async def run_concurrent_session_turn(
    retriever,
    *,
    raw_query: str,
    search_type_for_spans: SearchType | None = None,
) -> tuple[Any, Any, Any]:
    """Run one concurrent turn. Caller must already know the call qualifies.

    Returns ``(retrieved_objects, context, completion)``.
    """
    user = session_user.get()
    user_uuid = user.id
    session_manager = get_session_manager()
    user_cache_key = str(user_uuid)
    session_id = session_manager.resolve_session_id(retriever.session_id)

    async with session_turn_lock(user_cache_key, session_id):
        snapshot = await load_turn_context(
            session_manager,
            user_id=user_cache_key,
            session_id=session_id,
            raw_message=raw_query,
        )
        answer_lane = _retrieve_and_answer(
            retriever,
            raw_query=raw_query,
            snapshot=snapshot,
            user_uuid=user_uuid,
            session_id=session_id,
            search_type_for_spans=search_type_for_spans,
        )
        if session_manager.is_auto_feedback_enabled():
            analysis, answer_lane_result = await asyncio.gather(
                analyze_turn(snapshot),
                answer_lane,
            )
        else:
            analysis = SessionTurnAnalysis()
            answer_lane_result = await answer_lane
        retrieved_objects, context, answer = answer_lane_result

        await commit_turn(
            session_manager,
            snapshot=snapshot,
            analysis=analysis,
            answer=answer,
            user_id=user_cache_key,
            session_id=session_id,
            used_graph_element_ids=retriever.extract_context_object_ids(retrieved_objects),
        )

    completions = await retriever.append_references([answer], retrieved_objects)
    return retrieved_objects, context, completions


async def run_sequential_session_turn(
    retriever,
    *,
    raw_query: str,
    only_context: bool = False,
    search_type_for_spans: SearchType | None = None,
) -> tuple[Any, Any, Any]:
    """Prepare → objects → context → completion (the ``dev`` search path).

    Returns ``(retrieved_objects, context, completion)``.
    """
    retriever_class = _retriever_class_name(retriever)
    effective_query = raw_query
    turn_preparation = None

    if not only_context and getattr(retriever, "supports_session_turn_preparation", True):
        turn_preparation = await retriever.prepare_session_turn_for_retrieval(raw_query)
        if not turn_preparation.should_answer:
            return None, None, [turn_preparation.response_to_user or "Got it."]
        effective_query = turn_preparation.effective_query or raw_query

    with new_span("cognee.retrieval.get_objects") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        if search_type_for_spans is not None:
            span.set_attribute(COGNEE_SEARCH_TYPE, search_type_for_spans.value)
        retrieved_objects = await retriever.get_retrieved_objects(query=effective_query)
        obj_count = count_retrieved_objects(retrieved_objects)
        span.set_attribute(COGNEE_RESULT_COUNT, obj_count)
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} retrieved {obj_count} object(s)",
        )

    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)

    with new_span("cognee.retrieval.get_context") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        context = await retriever.get_context_from_objects(
            query=effective_query, retrieved_objects=retrieved_objects
        )
        if isinstance(context, str):
            span.set_attribute("cognee.retrieval.context_length", len(context))
        elif isinstance(context, list):
            span.set_attribute("cognee.retrieval.context_items", len(context))

    if only_context:
        return retrieved_objects, context, None

    with new_span("cognee.retrieval.get_completion") as span:
        span.set_attribute("cognee.retrieval.retriever", retriever_class)
        completion_kwargs = {
            "query": raw_query,
            "retrieved_objects": retrieved_objects,
            "context": context,
        }
        completion_method = retriever.get_completion_from_context
        if _method_accepts_kwarg(completion_method, "effective_query"):
            completion_kwargs["effective_query"] = effective_query
        if _method_accepts_kwarg(completion_method, "turn_preparation"):
            completion_kwargs["turn_preparation"] = turn_preparation
        completion = await completion_method(**completion_kwargs)
        if isinstance(completion, str):
            span.set_attribute("cognee.retrieval.completion_length", len(completion))
        span.set_attribute(
            COGNEE_RESULT_SUMMARY,
            f"{retriever_class} generated completion",
        )

    return retrieved_objects, context, completion


async def run_session_aware_completion(
    retriever,
    *,
    raw_query: str,
    original_search_type: SearchType | None = None,
    only_context: bool = False,
    is_batch: bool = False,
    search_type_for_spans: SearchType | None = None,
) -> tuple[Any, Any, Any]:
    """Only public door: concurrent or sequential, same ``(objects, context, completion)``."""
    if should_run_concurrent(
        retriever,
        original_search_type=original_search_type,
        is_batch=is_batch,
        only_context=only_context,
    ):
        return await run_concurrent_session_turn(
            retriever,
            raw_query=raw_query,
            search_type_for_spans=search_type_for_spans,
        )
    return await run_sequential_session_turn(
        retriever,
        raw_query=raw_query,
        only_context=only_context,
        search_type_for_spans=search_type_for_spans,
    )
