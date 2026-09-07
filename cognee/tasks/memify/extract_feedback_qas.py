from typing import Any, List, Optional

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.cache import SessionQAEntry
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.feedback_weights_constants import (
    FEEDBACK_SOURCE_EXPLICIT,
    FEEDBACK_SOURCE_IMPLICIT,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)

logger = get_logger("extract_feedback_qas")


def _valid_rating(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 1 <= value <= 5 else None


def build_implicit_rating_map(context_rows: Any) -> dict[str, tuple[int, str]]:
    """Map qa_id -> (rating, feedback text) from stored feedback rows, latest row winning.

    A feedback row is the turn-analysis record of the user's *next* message: its
    ``referenced_qa_rating`` is the 1-5 verdict inferred for the answer named in
    ``referenced_qa_ids`` and its ``raw_text`` is what the user actually said.
    """
    ratings: dict[str, tuple[int, str]] = {}
    for raw in context_rows if isinstance(context_rows, list) else []:
        row = raw if isinstance(raw, dict) else getattr(raw, "__dict__", None)
        if not isinstance(row, dict) or row.get("kind") != "feedback":
            continue
        rating = _valid_rating(row.get("referenced_qa_rating"))
        if rating is None:
            continue
        raw_text = row.get("raw_text")
        feedback_text = raw_text.strip() if isinstance(raw_text, str) else ""
        for qa_id in row.get("referenced_qa_ids") or []:
            if isinstance(qa_id, str) and qa_id:
                ratings[qa_id] = (rating, feedback_text)
    return ratings


def resolve_feedback(
    entry: SessionQAEntry,
    implicit_ratings: dict[str, tuple[int, str]],
) -> Optional[tuple[int, str, Optional[str]]]:
    """Return (rating, source, feedback_text) for a QA entry, or None when it carries no rating.

    An explicit ``feedback_score`` always wins; the implicit rating inferred from the
    user's next turn is used only when no explicit score exists.
    """
    explicit = _valid_rating(entry.feedback_score)
    if explicit is not None:
        return explicit, FEEDBACK_SOURCE_EXPLICIT, entry.feedback_text

    qa_id = entry.qa_id
    if isinstance(qa_id, str) and qa_id in implicit_ratings:
        rating, feedback_text = implicit_ratings[qa_id]
        return rating, FEEDBACK_SOURCE_IMPLICIT, feedback_text or None

    return None


def _is_already_applied(entry: SessionQAEntry) -> bool:
    memify_metadata = entry.memify_metadata
    return (
        isinstance(memify_metadata, dict)
        and memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY) is True
    )


async def extract_feedback_qas(data, session_ids: Optional[List[str]] = None):
    """
    Read provided sessions and yield rated QAs not yet applied to graph weights.

    Rows without ``used_graph_element_ids`` are yielded too, so the weight task can mark
    them processed once and they are never rescanned.
    """
    if (
        not isinstance(session_ids, list)
        or not session_ids
        or any(not isinstance(session_id, str) or not session_id for session_id in session_ids)
    ):
        raise CogneeValidationError(
            message="session_ids must be provided for extract_feedback_qas",
            log=False,
        )

    if not data or data == [{}]:
        logger.info("Extracting feedback QAs from session cache")

    user: User = session_user.get()
    if not user:
        raise CogneeSystemError(message="No authenticated user found in context", log=False)

    session_manager = get_session_manager()

    user_id = str(user.id)

    for session_id in session_ids:
        context_rows = await session_manager.get_session_context_entries(
            user_id=user_id,
            session_id=session_id,
        )
        implicit_ratings = build_implicit_rating_map(context_rows)

        entries = await session_manager.get_session(
            user_id=user_id,
            session_id=session_id,
            formatted=False,
        )
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, SessionQAEntry) or _is_already_applied(entry):
                continue

            qa_id = entry.qa_id
            if not isinstance(qa_id, str) or not qa_id:
                continue

            feedback = resolve_feedback(entry, implicit_ratings)
            if feedback is None:
                continue
            feedback_score, feedback_source, feedback_text = feedback

            memify_metadata = entry.memify_metadata
            yield {
                "session_id": session_id,
                "qa_id": qa_id,
                "feedback_score": feedback_score,
                "feedback_source": feedback_source,
                "feedback_text": feedback_text,
                "used_graph_element_ids": entry.used_graph_element_ids,
                "memify_metadata": memify_metadata if isinstance(memify_metadata, dict) else {},
            }
