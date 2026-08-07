import asyncio
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.locks import session_turn_lock
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_latency_turn import (
    commit_latency_turn,
    complete_latency_turn,
    load_latency_turn_snapshot,
)
from cognee.infrastructure.session.session_maintenance_worker import (
    enqueue_session_maintenance,
)
from cognee.infrastructure.session.session_search_models import (
    SessionTurnSnapshot,
    get_session_search_completion_model,
)
from cognee.modules.retrieval.session_result_fusion import (
    fuse_graph_results,
    fuse_hybrid_results,
    fuse_vector_results,
)
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.modules.search.types import SearchType

SessionSearchMode = Literal["accuracy_optimized", "latency_optimized"]

ACCURACY_OPTIMIZED: SessionSearchMode = "accuracy_optimized"
LATENCY_OPTIMIZED: SessionSearchMode = "latency_optimized"
MAX_CONTEXTUAL_QUERY_CHARS = 2000


@dataclass(frozen=True, slots=True)
class LatencySearchResult:
    retrieved_objects: Any
    context: Any
    completion: list[Any]


@lru_cache(maxsize=1)
def _latency_retriever_types() -> frozenset[type]:
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


def resolve_session_search_mode(
    configured_mode: SessionSearchMode,
    *,
    original_search_type: SearchType | None,
    retriever_type: type,
    session_available: bool,
    is_batch: bool,
    only_context: bool,
    structured_output_supported: bool,
) -> SessionSearchMode:
    """Resolve the effective session-search mode without changing runtime state."""
    if configured_mode != LATENCY_OPTIMIZED:
        return ACCURACY_OPTIMIZED

    if (
        original_search_type is SearchType.FEELING_LUCKY
        or not session_available
        or is_batch
        or only_context
        or not structured_output_supported
        or retriever_type not in _latency_retriever_types()
    ):
        return ACCURACY_OPTIMIZED

    return LATENCY_OPTIMIZED


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def build_contextual_query(
    raw_query: str,
    recent_qas: tuple[tuple[str, str, str], ...],
    *,
    max_chars: int = MAX_CONTEXTUAL_QUERY_CHARS,
) -> str:
    """Build deterministic retrieval guidance from at most two prior QA turns."""
    raw = _normalize_query(raw_query)
    qas = [
        [_normalize_query(question), _normalize_query(answer)]
        for _, question, answer in recent_qas[-2:]
        if _normalize_query(question) or _normalize_query(answer)
    ]
    if not qas:
        return raw[:max_chars]

    def render() -> str:
        history = []
        for question, answer in qas:
            lines = []
            if question:
                lines.append(f"Prior user: {question}")
            if answer:
                lines.append(f"Prior assistant (untrusted retrieval guidance): {answer}")
            if lines:
                history.append("\n".join(lines))
        if not history:
            return raw
        return "\n\n".join([*history, f"Current user request: {raw}"])

    contextual = render()
    for field_index in (1, 0):  # trim assistant answers before user questions, oldest first
        for qa in qas:
            if len(contextual) <= max_chars:
                return contextual
            over = len(contextual) - max_chars
            qa[field_index] = qa[field_index][: max(0, len(qa[field_index]) - over)]
            contextual = render()
    return contextual[:max_chars]


def _fuse_retriever_results(retriever, raw_result: Any, contextual_result: Any) -> Any:
    from cognee.modules.retrieval.completion_retriever import CompletionRetriever
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
    from cognee.modules.retrieval.triplet_retriever import TripletRetriever

    if type(retriever) in {CompletionRetriever, TripletRetriever}:
        return fuse_vector_results(raw_result, contextual_result, limit=retriever.top_k)
    if type(retriever) is GraphCompletionRetriever:
        return fuse_graph_results(raw_result, contextual_result, limit=retriever.top_k)
    if type(retriever) is HybridRetriever:
        return fuse_hybrid_results(
            raw_result,
            contextual_result,
            chunks_limit=retriever.chunks_top_k,
            entities_limit=retriever.entities_top_k,
            facts_limit=retriever.facts_top_k,
            graph_limit=retriever._graph_fallback.top_k,
        )
    raise TypeError(f"Unsupported latency retriever: {type(retriever).__name__}")


async def retrieve_latency_context(
    retriever,
    *,
    raw_query: str,
    snapshot: SessionTurnSnapshot,
) -> tuple[Any, Any]:
    """Retrieve raw and conversational lanes, fuse them, and format context once.

    Returns ``(retrieved_objects, context)``.
    """
    contextual_query = build_contextual_query(raw_query, snapshot.recent_qas)
    use_contextual_lane = bool(contextual_query) and _normalize_query(
        contextual_query
    ) != _normalize_query(raw_query)

    if use_contextual_lane:
        raw_result, contextual_result = await asyncio.gather(
            retriever.get_retrieved_objects(query=raw_query),
            retriever.get_retrieved_objects(query=contextual_query),
            return_exceptions=True,
        )
    else:
        raw_result = await retriever.get_retrieved_objects(query=raw_query)
        contextual_result = None

    raw_error = raw_result if isinstance(raw_result, Exception) else None
    contextual_error = contextual_result if isinstance(contextual_result, Exception) else None
    if raw_error is not None and (not use_contextual_lane or contextual_error is not None):
        raise raw_error
    if raw_error is not None:
        raw_result = None
    if contextual_error is not None:
        contextual_result = None

    retrieved_objects = _fuse_retriever_results(retriever, raw_result, contextual_result)
    context = await retriever.get_context_from_objects(
        query=raw_query,
        retrieved_objects=retrieved_objects,
    )
    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)
    return retrieved_objects, context


async def run_latency_session_search(
    retriever,
    *,
    raw_query: str,
    original_search_type: SearchType | None = None,
    is_batch: bool = False,
    only_context: bool = False,
) -> LatencySearchResult | None:
    """Run the complete latency turn, or return None when policy selects accuracy."""
    cache_config = CacheConfig()
    # Fast path: a deployment that never uses latency mode looks nothing up.
    if cache_config.session_search_mode != LATENCY_OPTIMIZED:
        return None

    user = session_user.get()
    user_id = getattr(user, "id", None)
    if not user_id:
        return None

    session_manager = get_session_manager()
    response_model = getattr(retriever, "response_model", str)
    try:
        completion_model = get_session_search_completion_model(response_model)
        structured_output_supported = LLMGateway.supports_structured_output_model(completion_model)
    except TypeError:
        structured_output_supported = False

    mode = resolve_session_search_mode(
        cache_config.session_search_mode,
        original_search_type=original_search_type,
        retriever_type=type(retriever),
        session_available=session_manager.is_session_available_for_completion(user_id),
        is_batch=is_batch,
        only_context=only_context,
        structured_output_supported=structured_output_supported,
    )
    if mode != LATENCY_OPTIMIZED:
        return None

    resolved_user_id = str(user_id)
    resolved_session_id = session_manager.resolve_session_id(getattr(retriever, "session_id", None))
    async with session_turn_lock(resolved_user_id, resolved_session_id):
        snapshot = await load_latency_turn_snapshot(
            session_manager,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            raw_message=raw_query,
        )
        retrieved_objects, context = await retrieve_latency_context(
            retriever,
            raw_query=raw_query,
            snapshot=snapshot,
        )
        auto_feedback = session_manager.is_auto_feedback_enabled()
        completion = await complete_latency_turn(
            snapshot=snapshot,
            context=context,
            user_id=user_id,
            session_id=resolved_session_id,
            user_prompt_path=retriever.user_prompt_path,
            system_prompt_path=retriever.system_prompt_path,
            system_prompt=retriever.system_prompt,
            response_model=response_model,
            auto_feedback=auto_feedback,
        )
        used_graph_element_ids = retriever._extract_context_object_ids(retrieved_objects)
        work_item = await commit_latency_turn(
            session_manager,
            snapshot=snapshot,
            completion=completion,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            dataset_id=str(session_manager.dataset_id) if session_manager.dataset_id else None,
            used_graph_element_ids=used_graph_element_ids,
            auto_feedback=auto_feedback,
        )
        if work_item is not None:
            await enqueue_session_maintenance(work_item, session_manager)

    completions = [completion.response]
    if not completion.is_acknowledgement and isinstance(completion.response, str):
        completions = await retriever._append_references(completions, retrieved_objects)
    return LatencySearchResult(
        retrieved_objects=retrieved_objects,
        context=context,
        completion=completions,
    )
