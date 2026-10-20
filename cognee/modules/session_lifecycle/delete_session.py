"""Explicit deletion of one session: everything cognee keeps for it, owner only.

A session lives in two places. The session cache holds its content -- Q&A
turns, agent-trace steps, the session-context list (which also carries the
improve() watermarks) -- plus the Q&A vectors next to it; the relational DB
holds its dashboard row (``session_records``) and per-model usage rows
(``session_model_usage``). ``delete_session`` removes both, so the session
disappears from recall, from improve(), and from the sessions API alike.

Not touched, by design: lessons already distilled from the session into a
dataset (ordinary documents, removed with ``forget(data_id=...)``), and
``pipeline_runs`` rows that name the session (run history).
"""

from uuid import UUID

from sqlalchemy import delete

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.locks import session_turn_lock
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger

from .models import SessionModelUsage, SessionRecord

logger = get_logger("delete_session")


async def _delete_session_rows(user_id: UUID, session_id: str) -> int:
    """Delete the session's dashboard and usage rows; return how many rows went."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        deleted = 0
        for model in (SessionModelUsage, SessionRecord):
            result = await db_session.execute(
                delete(model).where(model.user_id == user_id, model.session_id == session_id)
            )
            deleted += result.rowcount or 0
        await db_session.commit()
    return deleted


async def delete_session(user_id: UUID, session_id: str) -> bool:
    """Delete one of ``user_id``'s sessions; True if anything existed, False if nothing did.

    Raises when session memory is off (``CACHING=false``): there is no cache
    to delete from, and answering False would read as "no such session".
    Cache errors propagate. Holds the session's turn lock, so an answer being
    written to the same session finishes before the delete, never after it.
    """
    if not session_id:
        raise CogneeValidationError(
            message="delete_session needs an explicit session_id.",
            name="SessionPreconditionError",
        )
    session_manager = get_session_manager()
    if not session_manager.is_available:
        raise CogneeValidationError(
            message=(
                "Session memory is disabled (CACHING=false), so there is no session to delete. "
                "Enable CACHING to use sessions."
            ),
            name="SessionPreconditionError",
        )

    async with session_turn_lock(str(user_id), session_id):
        cache_deleted = await session_manager.delete_session(
            user_id=str(user_id), session_id=session_id
        )
        rows_deleted = await _delete_session_rows(user_id, session_id)

    logger.info(
        "Deleted session %s of user %s: cache content %s, %d relational row(s).",
        session_id,
        user_id,
        "removed" if cache_deleted else "absent",
        rows_deleted,
    )
    return cache_deleted or rows_deleted > 0
