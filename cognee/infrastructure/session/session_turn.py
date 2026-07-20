"""Processing one session turn: interpret the user message, apply its feedback to the
stored session context, and assemble the prompt for the answer.

These are the helpers behind SessionManager's turn flow. Like ``session_context_builder``,
they take the SessionManager as a parameter and call back into its storage facade, so
SessionManager stays an orchestrator plus a thin facade rather than holding this logic.
All public coroutines are fail-open so they never block answer generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cognee.context_global_variables import session_user
from cognee.infrastructure.session.feedback_detection import analyze_turn_for_session_context
from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_context_builder import (
    apply_candidate_updates,
    build_active_context_block,
)
from cognee.infrastructure.session.session_context_models import SessionFeedbackEntry
from cognee.infrastructure.session.session_embeddings import (
    merge_hybrid_qa_entries,
    search_session_qa_ids,
)
from cognee.modules.retrieval.utils.completion import (
    generate_session_completion_with_optional_summary,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_turn")


@dataclass
class SessionTurnPreparation:
    """Pre-answer decision and updates for one session turn."""

    should_answer: bool = True
    response_to_user: str | None = None
    effective_query: str = ""
    analysis: SessionTurnAnalysis | None = None
    accepted_context_ids: list[str] = field(default_factory=list)
    previous_qa_id: str | None = None


def compose_session_prompt(
    active_context_block: str,
    conversation_history: str,
) -> str:
    """Assemble the session prompt from active guidance and conversation history.

    Empty layers are skipped. The active session-context block is placed before
    the conversation history so durable user/session guidance remains prominent.
    """
    prompt = conversation_history
    if active_context_block:
        prompt = active_context_block + "\n\n" + prompt
    return prompt


def _empty_turn_preparation(query: str) -> SessionTurnPreparation:
    return SessionTurnPreparation(should_answer=True, effective_query=query)


def coerce_qa_entry(entry: Any) -> dict:
    """Normalize a stored QA entry (model or dict) to a plain dict."""
    if hasattr(entry, "model_dump"):
        return entry.model_dump()
    if isinstance(entry, dict):
        return entry
    return {}


async def select_session_history(
    session_manager,
    user_id: str,
    session_id: str,
    query_text: str,
) -> str:
    """Load session history and return it as a formatted conversation string.

    History is the union of the last N turns and vector-engine hits, in
    chronological order. On any failure, this falls back to the plain last-N window.
    """
    try:
        storage_id_resolver = getattr(session_manager, "get_storage_session_id", None)
        storage_session_id = (
            storage_id_resolver(session_id) if storage_id_resolver is not None else session_id
        )
        vector_qa_ids = await search_session_qa_ids(
            user_id=user_id,
            session_id=storage_session_id,
            query_text=query_text,
        )
        recent_entries = await session_manager.get_session(
            user_id=user_id,
            session_id=session_id,
            formatted=False,
            last_n=session_manager.session_history_last_n,
        )
        vector_entries = []
        if vector_qa_ids:
            vector_entries = await session_manager.get_session_entries_by_ids(
                user_id=user_id,
                session_id=session_id,
                qa_ids=vector_qa_ids,
            )

        if isinstance(recent_entries, list) and isinstance(vector_entries, list):
            selected = merge_hybrid_qa_entries(recent_entries, vector_entries)
            return session_manager.format_entries(
                [coerce_qa_entry(entry) for entry in selected],
                include_context=False,
            )
    except Exception as error:
        logger.warning("Session history: hybrid selection failed open: %s", error)

    history = await session_manager.get_session(
        user_id=user_id,
        session_id=session_id,
        formatted=True,
        last_n=session_manager.session_history_last_n,
        include_context=False,
    )
    return history if isinstance(history, str) else ""


async def generate_session_answer(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    answer_query: str,
    context: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None,
    response_model: type,
    summarize_context: bool,
    max_context_chars: int | None,
) -> tuple[Any, str, list[str] | None]:
    """Recall history and context, compose the prompt, and generate one answer.

    Returns ``(answer, context_to_store, served_context_ids)``.
    """
    conversation_history = await select_session_history(
        session_manager,
        user_id,
        session_id,
        query_text=answer_query,
    )

    served_ids: list[str] = []
    active_context_block = ""
    if session_manager.is_auto_feedback_enabled():
        active_context_block, served_ids = await build_active_context_block_safe(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            query=answer_query,
        )

    conversation_history = compose_session_prompt(active_context_block, conversation_history)

    (
        answer,
        context_to_store,
        _feedback_result,
    ) = await generate_session_completion_with_optional_summary(
        query=answer_query,
        context=context,
        conversation_history=conversation_history,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
        response_model=response_model,
        summarize_context=summarize_context,
    )
    return answer, context_to_store, served_ids or None


async def build_active_context_block_safe(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    query: str,
) -> tuple[str, list[str]]:
    """Render the active session-context guidance block. Fail-open -> ("", [])."""
    try:
        return await build_active_context_block(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            query=query,
        )
    except Exception as e:
        logger.warning("Active session-context block failed: %s", e)
        return "", []


async def load_served_context_payload(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    served_ids: list[str],
) -> list[dict]:
    """Resolve the context entries served to the previous answer into {id, content} dicts.

    These feed the single turn-analysis call so it can rate them. Fail-open -> [].
    """
    if not served_ids:
        return []
    try:
        entries = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
        by_id = {}
        for raw in entries or []:
            row = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {})
            entry_id = row.get("id")
            if entry_id is not None and row.get("kind", "context") == "context":
                by_id[str(entry_id)] = row.get("content", "")
        return [{"id": cid, "content": by_id[cid]} for cid in served_ids if cid in by_id]
    except Exception as e:
        logger.warning("Session turn: load served context failed: %s", e)
        return []


async def apply_served_context_ratings(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    ratings: list,
) -> None:
    """Increment helpful_count / harmful_count for rated entries. Fail-open per rating."""
    if not ratings:
        return
    try:
        entries = await session_manager.get_session_context_entries(
            user_id=user_id, session_id=session_id
        )
        counts = {}
        for raw in entries or []:
            row = raw if isinstance(raw, dict) else getattr(raw, "__dict__", {})
            if row.get("kind", "context") != "context":
                continue
            entry_id = row.get("id")
            if entry_id is not None:
                counts[str(entry_id)] = (
                    int(row.get("helpful_count", 0) or 0),
                    int(row.get("harmful_count", 0) or 0),
                )
        for rating in ratings:
            try:
                entry_id = str(getattr(rating, "entry_id", None) or "")
                verdict = str(getattr(rating, "rating", "") or "").strip().lower()
                if entry_id not in counts or verdict not in ("helpful", "harmful"):
                    continue
                helpful, harmful = counts[entry_id]
                if verdict == "helpful":
                    merge = {"helpful_count": helpful + 1}
                    next_counts = (helpful + 1, harmful)
                else:
                    merge = {"harmful_count": harmful + 1}
                    next_counts = (helpful, harmful + 1)
                await session_manager.update_session_context_entry(
                    user_id=user_id,
                    entry_id=entry_id,
                    merge=merge,
                    session_id=session_id,
                )
                counts[entry_id] = next_counts
            except Exception:
                continue
    except Exception as e:
        logger.warning("Session turn: served-context rating update failed: %s", e)


async def apply_session_turn_analysis(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    query: str,
    analysis: SessionTurnAnalysis,
    previous_qa_id: str | None,
    served_ids: list[str],
) -> list[str]:
    """Persist turn evidence, apply candidate updates, and bump helpful/harmful counters."""
    if not analysis.candidate_context_updates and not analysis.served_context_ratings:
        return []
    try:
        ratings = list(analysis.served_context_ratings or [])
        candidates = list(analysis.candidate_context_updates or [])

        feedback_entry = SessionFeedbackEntry(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            raw_text=query,
            referenced_qa_ids=[previous_qa_id] if previous_qa_id else [],
            influencing_context_ids=list(served_ids or []),
            candidate_context_entries=[
                c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in candidates
            ],
        )
        await session_manager.create_session_context_entry(
            user_id=user_id,
            entry_dump=feedback_entry.model_dump(),
            session_id=session_id,
        )

        touched_ids = await apply_candidate_updates(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            source_id=feedback_entry.id,
            candidates=candidates,
        )

        await apply_served_context_ratings(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            ratings=ratings,
        )
        return touched_ids
    except Exception as e:
        logger.warning("Session turn: feedback application failed: %s", e)
        return []


async def prepare_session_turn(
    session_manager,
    *,
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> SessionTurnPreparation:
    """Analyze one user turn before retrieval/answer generation.

    Runs only when caching and auto_feedback are enabled. Applies accepted candidate
    guidance, rates previously served guidance, and returns the effective query that
    retrieval and answer generation should use. Fail-open to a pass-through on any error.
    """
    resolved_user_id = user_id
    if resolved_user_id is None:
        user = session_user.get()
        resolved_user_id = getattr(user, "id", None)

    if not session_manager.is_session_available_for_completion(resolved_user_id):
        return _empty_turn_preparation(query)
    if not session_manager.is_auto_feedback_enabled():
        return _empty_turn_preparation(query)

    resolved_session_id = session_manager._resolve_session_id(session_id)

    try:
        previous_entries = await session_manager.get_session(
            user_id=str(resolved_user_id),
            session_id=resolved_session_id,
            formatted=False,
            last_n=1,
        )
        previous_entry = (
            coerce_qa_entry(previous_entries[-1])
            if isinstance(previous_entries, list) and previous_entries
            else {}
        )
        previous_qa_id = previous_entry.get("qa_id")
        previous_question = previous_entry.get("question")
        previous_answer = previous_entry.get("answer")
        previous_served_ids = previous_entry.get("used_session_context_ids") or []
        if not isinstance(previous_served_ids, list):
            previous_served_ids = []

        served_context = await load_served_context_payload(
            session_manager,
            user_id=str(resolved_user_id),
            session_id=resolved_session_id,
            served_ids=[str(entry_id) for entry_id in previous_served_ids],
        )

        analysis = await analyze_turn_for_session_context(
            query,
            previous_question=previous_question,
            previous_answer=previous_answer,
            served_context=served_context,
        )
    except Exception as error:
        logger.warning("Session turn preparation failed open: %s", error)
        return _empty_turn_preparation(query)

    try:
        accepted_context_ids = await apply_session_turn_analysis(
            session_manager,
            user_id=str(resolved_user_id),
            session_id=resolved_session_id,
            query=query,
            analysis=analysis,
            previous_qa_id=previous_qa_id,
            served_ids=[str(entry_id) for entry_id in previous_served_ids],
        )
    except Exception as error:
        logger.warning("Session turn analysis application failed open: %s", error)
        accepted_context_ids = []

    query_to_answer = (analysis.query_to_answer or "").strip()
    response_to_user = (analysis.response_to_user or "").strip() or None
    has_analysis_signal = bool(
        query_to_answer
        or response_to_user
        or analysis.candidate_context_updates
        or analysis.served_context_ratings
    )
    has_previous_answer = bool(previous_qa_id)
    should_answer = bool(query_to_answer or not has_analysis_signal or not has_previous_answer)
    effective_query = query_to_answer or query
    if not should_answer and not response_to_user:
        response_to_user = "Got it."

    return SessionTurnPreparation(
        should_answer=should_answer,
        response_to_user=response_to_user,
        effective_query=effective_query,
        analysis=analysis,
        accepted_context_ids=accepted_context_ids,
        previous_qa_id=previous_qa_id,
    )
