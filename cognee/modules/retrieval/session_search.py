import asyncio
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.locks import session_turn_lock
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_concurrent_turn import (
    analyze_turn_concurrently,
    commit_turn,
    complete_turn,
    load_turn_snapshot,
)
from cognee.infrastructure.session.session_search_models import SessionTurnSnapshot
from cognee.modules.retrieval.utils.access_tracking import update_node_access_timestamps
from cognee.modules.search.types import SearchType

CONCURRENT_MODE = "concurrent"
MAX_CONTEXTUAL_QUERY_CHARS = 2000


@dataclass(frozen=True, slots=True)
class ConcurrentTurnResult:
    retrieved_objects: Any
    context: Any
    completion: list[Any]


@lru_cache(maxsize=1)
def _eligible_retriever_types() -> frozenset[type]:
    """Exact concrete classes that support a concurrent turn.

    Opt-in is by exact type, never by subclass: a variant that overrides retrieval or
    runs its own LLM rounds (chain-of-thought, context extension) cannot honour the
    one-blocking-completion contract, and inherits its way onto this list otherwise.
    """
    from cognee.modules.retrieval.completion_retriever import CompletionRetriever
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.graph_summary_completion_retriever import (
        GraphSummaryCompletionRetriever,
    )
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
    from cognee.modules.retrieval.triplet_retriever import TripletRetriever

    return frozenset(
        {
            CompletionRetriever,
            GraphCompletionRetriever,
            GraphSummaryCompletionRetriever,
            HybridRetriever,
            TripletRetriever,
        }
    )


def can_run_as_concurrent_turn(
    *,
    original_search_type: SearchType | None,
    retriever_type: type,
    session_available: bool,
    is_batch: bool,
    only_context: bool,
) -> bool:
    """Whether this call can run as one concurrent turn, given concurrent mode is configured.

    Everything else falls back to the sequential path, which is unchanged.
    """
    return not (
        original_search_type is SearchType.FEELING_LUCKY
        or not session_available
        or is_batch
        or only_context
        or retriever_type not in _eligible_retriever_types()
    )


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


async def retrieve_turn_context(
    retriever,
    *,
    raw_query: str,
    snapshot: SessionTurnSnapshot,
) -> tuple[Any, Any]:
    """Retrieve raw and conversational lanes, fuse them, and format context once.

    Returns ``(retrieved_objects, context)``.
    """
    contextual_query = build_contextual_query(raw_query, snapshot.recent_qas)
    # With no prior turns the builder returns the raw query itself, so there is one lane.
    use_contextual_lane = bool(contextual_query) and contextual_query != _normalize_query(raw_query)

    if use_contextual_lane:
        raw_result, contextual_result = await asyncio.gather(
            retriever.get_retrieved_objects(query=raw_query),
            retriever.get_retrieved_objects(query=contextual_query),
            return_exceptions=True,
        )
    else:
        raw_result = await retriever.get_retrieved_objects(query=raw_query)
        contextual_result = None

    if isinstance(raw_result, Exception):
        if not use_contextual_lane or isinstance(contextual_result, Exception):
            raise raw_result
        raw_result = None
    if isinstance(contextual_result, Exception):
        contextual_result = None

    retrieved_objects = retriever.merge_retrieved_objects(raw_result, contextual_result)
    context = await retriever.get_context_from_objects(
        query=raw_query,
        retrieved_objects=retrieved_objects,
    )
    if retrieved_objects:
        await update_node_access_timestamps(retrieved_objects)
    return retrieved_objects, context


async def _retrieve_and_answer(
    retriever,
    *,
    raw_query: str,
    snapshot: SessionTurnSnapshot,
    user_id: Any,
    session_id: str,
) -> tuple[Any, Any, Any]:
    """The answer lane: retrieve both queries, merge, format, and answer once."""
    retrieved_objects, context = await retrieve_turn_context(
        retriever,
        raw_query=raw_query,
        snapshot=snapshot,
    )
    answer = await complete_turn(
        snapshot=snapshot,
        context=context,
        user_id=user_id,
        session_id=session_id,
        user_prompt_path=retriever.user_prompt_path,
        system_prompt_path=retriever.system_prompt_path,
        system_prompt=retriever.system_prompt,
        response_model=retriever.response_model,
    )
    return retrieved_objects, context, answer


async def try_concurrent_turn(
    retriever,
    *,
    raw_query: str,
    original_search_type: SearchType | None = None,
    is_batch: bool = False,
    only_context: bool = False,
) -> ConcurrentTurnResult | None:
    """Run this search as one concurrent turn, or return None if it does not qualify.

    Returning None is the caller's signal to fall through to the sequential path, which is
    why this is the only integration point the rest of the search flow needs.

    The turn analysis and the answer are independent — the analysis reads the user's
    message against the previous turn, never this turn's answer — so they run
    concurrently and the turn costs one answer call of wall-clock time.
    """
    cache_config = CacheConfig()
    # Fast path: a deployment that never uses concurrent mode looks nothing up.
    if cache_config.session_search_mode != CONCURRENT_MODE:
        return None

    user = session_user.get()
    user_id = getattr(user, "id", None)
    if not user_id:
        return None

    session_manager = get_session_manager()
    if not can_run_as_concurrent_turn(
        original_search_type=original_search_type,
        retriever_type=type(retriever),
        session_available=session_manager.is_session_available_for_completion(user_id),
        is_batch=is_batch,
        only_context=only_context,
    ):
        return None

    resolved_user_id = str(user_id)
    resolved_session_id = session_manager.resolve_session_id(retriever.session_id)
    async with session_turn_lock(resolved_user_id, resolved_session_id):
        snapshot = await load_turn_snapshot(
            session_manager,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            raw_message=raw_query,
        )
        answer_lane = _retrieve_and_answer(
            retriever,
            raw_query=raw_query,
            snapshot=snapshot,
            user_id=user_id,
            session_id=resolved_session_id,
        )
        if session_manager.is_auto_feedback_enabled():
            analysis, answered = await asyncio.gather(
                analyze_turn_concurrently(snapshot),
                answer_lane,
            )
        else:
            analysis = SessionTurnAnalysis()
            answered = await answer_lane
        retrieved_objects, context, answer = answered

        await commit_turn(
            session_manager,
            snapshot=snapshot,
            analysis=analysis,
            answer=answer,
            user_id=resolved_user_id,
            session_id=resolved_session_id,
            used_graph_element_ids=retriever._extract_context_object_ids(retrieved_objects),
        )

    completions = [answer]
    if isinstance(answer, str):
        completions = await retriever._append_references(completions, retrieved_objects)
    return ConcurrentTurnResult(
        retrieved_objects=retrieved_objects,
        context=context,
        completion=completions,
    )
