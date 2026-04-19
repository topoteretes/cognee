"""Session-lifecycle operations — thin wrappers around SessionRecord.

All public functions are async and idempotent where possible:
* ``ensure_session`` upserts on first write.
* ``touch_session`` bumps ``last_activity_at`` — cheap UPDATE.
* ``accumulate_usage`` atomically adds tokens / cost.
* ``mark_ended`` transitions to a terminal status.

The ``abandoned`` transition is never written — it's computed at read
time via ``get_effective_status_sql`` against
``SESSION_ABANDON_AFTER_SECONDS`` (defaults to 30 min).
"""

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Sequence
from uuid import UUID as UUIDType

from sqlalchemy import and_, case, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import SessionRecord

logger = get_logger("session_lifecycle")


class SessionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"  # computed at read time, not stored


# 30 minutes by default; overridable via env var for tests.
def _abandon_after_seconds() -> int:
    raw = os.environ.get("SESSION_ABANDON_AFTER_SECONDS", "")
    try:
        return int(raw) if raw else 1800
    except ValueError:
        return 1800


async def ensure_session(
    *,
    session_id: str,
    user_id: UUIDType,
    dataset_id: Optional[UUIDType] = None,
) -> None:
    """Create the session row if it doesn't exist; no-op if it does.

    Safe to call on every write — the INSERT uses ON CONFLICT DO
    NOTHING (SQLite) or equivalent. Subsequent activity updates flow
    through ``touch_session`` / ``accumulate_usage``.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        now = datetime.now(timezone.utc)
        stmt = sqlite_insert(SessionRecord).values(
            session_id=session_id,
            user_id=user_id,
            dataset_id=dataset_id,
            status=SessionStatus.RUNNING.value,
            started_at=now,
            last_activity_at=now,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error_count=0,
        )
        # ON CONFLICT DO NOTHING is SQLite-specific; Postgres supports
        # the same shape. For other dialects, the SELECT-then-INSERT
        # fallback below handles it.
        try:
            stmt = stmt.on_conflict_do_nothing(index_elements=["session_id", "user_id"])
            await session.execute(stmt)
            await session.commit()
            return
        except Exception:
            await session.rollback()

        # Fallback path for non-SQLite/Postgres dialects.
        existing = await session.execute(
            select(SessionRecord).where(
                and_(
                    SessionRecord.session_id == session_id,
                    SessionRecord.user_id == user_id,
                )
            )
        )
        if existing.scalar_one_or_none() is not None:
            return

        session.add(
            SessionRecord(
                session_id=session_id,
                user_id=user_id,
                dataset_id=dataset_id,
                status=SessionStatus.RUNNING.value,
                started_at=now,
                last_activity_at=now,
            )
        )
        await session.commit()


async def touch_session(
    *,
    session_id: str,
    user_id: UUIDType,
    dataset_id: Optional[UUIDType] = None,
) -> None:
    """Bump ``last_activity_at`` to now.

    Called after every cache write. Also opportunistically fills in
    ``dataset_id`` if it's currently null and we have one now.
    """
    now = datetime.now(timezone.utc)

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        # Only touch rows that are still active — terminal sessions
        # stay terminal. This avoids a late tool-call resurrecting a
        # session that SessionEnd already marked completed.
        stmt = (
            update(SessionRecord)
            .where(
                and_(
                    SessionRecord.session_id == session_id,
                    SessionRecord.user_id == user_id,
                    SessionRecord.status == SessionStatus.RUNNING.value,
                )
            )
            .values(last_activity_at=now)
        )
        if dataset_id is not None:
            # Don't clobber a pre-existing dataset_id — only fill null.
            stmt = stmt.values(
                dataset_id=case(
                    (SessionRecord.dataset_id.is_(None), dataset_id),
                    else_=SessionRecord.dataset_id,
                )
            )
        await session.execute(stmt)
        await session.commit()


async def accumulate_usage(
    *,
    session_id: str,
    user_id: UUIDType,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    model: Optional[str] = None,
    errored: bool = False,
) -> None:
    """Atomically add usage counters to the session row.

    Uses SQL-level arithmetic (column + :delta) so concurrent
    accumulators from two LLM calls don't clobber each other.
    """
    if tokens_in == 0 and tokens_out == 0 and cost_usd == 0.0 and not errored and model is None:
        return

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        stmt = update(SessionRecord).where(
            and_(
                SessionRecord.session_id == session_id,
                SessionRecord.user_id == user_id,
            )
        )
        values = {}
        if tokens_in:
            values["tokens_in"] = SessionRecord.tokens_in + tokens_in
        if tokens_out:
            values["tokens_out"] = SessionRecord.tokens_out + tokens_out
        if cost_usd:
            values["cost_usd"] = SessionRecord.cost_usd + cost_usd
        if errored:
            values["error_count"] = SessionRecord.error_count + 1
        if model:
            values["last_model"] = model
        if not values:
            return
        stmt = stmt.values(**values)
        await session.execute(stmt)
        await session.commit()


async def mark_ended(
    *,
    session_id: str,
    user_id: UUIDType,
    status: SessionStatus,
) -> None:
    """Transition to a terminal status (completed / failed)."""
    if status == SessionStatus.RUNNING or status == SessionStatus.ABANDONED:
        raise ValueError(
            f"mark_ended requires a terminal status (completed/failed), got {status}"
        )

    now = datetime.now(timezone.utc)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        stmt = (
            update(SessionRecord)
            .where(
                and_(
                    SessionRecord.session_id == session_id,
                    SessionRecord.user_id == user_id,
                )
            )
            .values(status=status.value, ended_at=now)
        )
        await session.execute(stmt)
        await session.commit()


def get_effective_status_sql():
    """Return a SQL expression that evaluates to the effective status.

    Usage (in a SELECT):
        from sqlalchemy import select
        select(SessionRecord, get_effective_status_sql().label("effective_status"))

    Rule: if stored status is ``running`` AND last_activity_at is older
    than ``SESSION_ABANDON_AFTER_SECONDS`` → ``abandoned``. Otherwise
    use the stored status verbatim.

    The threshold is computed in Python and passed as a bound
    parameter, so this works uniformly on SQLite, Postgres, etc.
    """
    threshold_seconds = _abandon_after_seconds()
    threshold_ts = datetime.now(timezone.utc) - timedelta(seconds=threshold_seconds)
    return case(
        (
            and_(
                SessionRecord.status == SessionStatus.RUNNING.value,
                SessionRecord.last_activity_at < threshold_ts,
            ),
            SessionStatus.ABANDONED.value,
        ),
        else_=SessionRecord.status,
    )


async def get_session_row(
    *,
    session_id: str,
    user_id: UUIDType,
) -> Optional[SessionRecord]:
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            select(SessionRecord).where(
                and_(
                    SessionRecord.session_id == session_id,
                    SessionRecord.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()


async def list_session_rows(
    *,
    user_id: Optional[UUIDType] = None,
    since: Optional[datetime] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "last_activity_at",
    descending: bool = True,
) -> Sequence[SessionRecord]:
    """List sessions with pagination.

    status_filter accepts the effective status (including
    'abandoned') — the SQL predicate handles the abandoned-by-time
    inference.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        eff = get_effective_status_sql()
        stmt = select(SessionRecord, eff.label("effective_status"))
        if user_id is not None:
            stmt = stmt.where(SessionRecord.user_id == user_id)
        if since is not None:
            stmt = stmt.where(SessionRecord.last_activity_at >= since)
        if status_filter:
            stmt = stmt.where(eff == status_filter)

        # Sort column allow-list to avoid arbitrary SQL injection via
        # the order_by param.
        sortable = {
            "last_activity_at": SessionRecord.last_activity_at,
            "started_at": SessionRecord.started_at,
            "ended_at": SessionRecord.ended_at,
            "cost_usd": SessionRecord.cost_usd,
            "tokens_in": SessionRecord.tokens_in,
            "tokens_out": SessionRecord.tokens_out,
        }
        sort_col = sortable.get(order_by, SessionRecord.last_activity_at)
        stmt = stmt.order_by(sort_col.desc() if descending else sort_col.asc())
        stmt = stmt.limit(limit).offset(offset)

        result = await session.execute(stmt)
        rows = result.all()
        # Attach the effective status to each SessionRecord for the caller.
        out = []
        for row in rows:
            rec = row[0]
            setattr(rec, "effective_status", row[1])
            out.append(rec)
        return out
