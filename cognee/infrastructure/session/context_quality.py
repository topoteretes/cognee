"""Project-specific evidence gates and dataset-scoped guidance retention."""

import os
import re
from datetime import datetime, timezone
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("context_quality")
_GENERIC = re.compile(
    r"(?:parse|parsing|inspect|read|check).{0,45}(?:bash|read|grep|glob|tool).{0,25}output"
    r"|(?:bash|read|grep|glob) succeeded[.!]?$"
    r"|(?:use|call) (?:read|grep|glob) (?:to|for) "
    r"|tool (?:output|response) (?:may|can|is|contains)",
    re.IGNORECASE,
)


def supported_project_candidate(candidate, source_id) -> bool:
    support = (getattr(candidate, "reason", "") or getattr(candidate, "evidence", "")).strip()
    return bool(len(support) >= 12 and not _GENERIC.search(candidate.content))


def unused_entry(entry, cutoff: datetime) -> bool:
    if entry.context_profile != "agent":
        return False
    try:
        used = datetime.fromisoformat(entry.last_served_at or entry.created_at)
        if used.tzinfo is None:
            used = used.replace(tzinfo=timezone.utc)
        return used < cutoff
    except (TypeError, ValueError):
        return False


async def filter_unused_entries(session_manager, user_id, session_id, entries, *, persist=False):
    """Archive guidance unused through N newer sessions of the same user and dataset.

    Read-only recall filters without writing. Missing lifecycle attribution fails open;
    a session from another dataset/user must never age this guidance.
    """
    try:
        from sqlalchemy import select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.session_lifecycle.models import SessionRecord

        count = max(1, int(os.getenv("COGNEE_AGENT_CONTEXT_UNUSED_SESSIONS", "20")))
        uid = UUID(str(user_id))
        engine = get_relational_engine()
        async with engine.get_async_session() as db:
            record = await db.scalar(
                select(SessionRecord).where(
                    SessionRecord.user_id == uid,
                    SessionRecord.session_id == session_id,
                )
            )
            if record is None or record.dataset_id is None:
                return entries
            cutoff = await db.scalar(
                select(SessionRecord.started_at)
                .where(
                    SessionRecord.user_id == uid,
                    SessionRecord.dataset_id == record.dataset_id,
                    SessionRecord.session_id != session_id,
                )
                .order_by(SessionRecord.started_at.desc())
                .offset(count - 1)
                .limit(1)
            )
        if cutoff is None:
            return entries
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        kept = []
        for entry in entries:
            if not unused_entry(entry, cutoff):
                kept.append(entry)
            elif persist:
                await session_manager.update_session_context_entry(
                    user_id=user_id,
                    session_id=session_id,
                    entry_id=entry.id,
                    merge={"expired_at": datetime.now(timezone.utc).isoformat()},
                )
        return kept
    except Exception:
        logger.debug("Guidance retention could not resolve session attribution", exc_info=True)
        return entries
