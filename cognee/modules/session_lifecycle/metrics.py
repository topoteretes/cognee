"""Session-lifecycle operations — thin wrappers around SessionRecord.

All public functions are async and idempotent where possible:

* ``ensure_and_touch_session`` upserts the row and bumps
  ``last_activity_at`` in a single DB round trip.
* ``accumulate_usage`` atomically adds tokens / cost to the session row
  and, when a model is named, updates the ``session_model_usage``
  table so ``cost-by-model`` attributes mixed-model sessions correctly.
* ``mark_ended`` transitions to a terminal status.

Writes to ``running`` sessions only — terminal sessions (completed /
failed) stay frozen so late tool-calls don't resurrect or distort them.

The ``abandoned`` transition is never written — it's computed at read
time via ``get_effective_status_sql`` against
``SESSION_ABANDON_AFTER_SECONDS`` (defaults to 30 min).
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Sequence
from uuid import UUID as UUIDType

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import SessionModelUsage, SessionRecord

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


def _dialect_name(bind) -> str:
    try:
        return bind.dialect.name
    except Exception:
        return ""


async def ensure_and_touch_session(
    *,
    session_id: str,
    user_id: UUIDType,
    dataset_id: Optional[UUIDType] = None,
) -> None:
    """Upsert the session row in one round trip.

    Creates the row if absent (status=running). If present AND still
    running, bumps ``last_activity_at``. Terminal sessions are left
    untouched so a late straggler can't accidentally resurrect them.
    Also fills in ``dataset_id`` when currently null.
    """
    now = datetime.now(timezone.utc)
    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        bind = await session.connection()
        dialect = _dialect_name(bind)

        values = {
            "session_id": session_id,
            "user_id": user_id,
            "dataset_id": dataset_id,
            "status": SessionStatus.RUNNING.value,
            "started_at": now,
            "last_activity_at": now,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "error_count": 0,
        }

        if dialect in ("sqlite", "postgresql"):
            insert = sqlite_insert if dialect == "sqlite" else pg_insert
            stmt = insert(SessionRecord).values(**values)
            set_ = {"last_activity_at": now}
            # Back-fill a previously-unset dataset_id.
            set_["dataset_id"] = case(
                (SessionRecord.dataset_id.is_(None), dataset_id),
                else_=SessionRecord.dataset_id,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["session_id", "user_id"],
                set_=set_,
                where=SessionRecord.status == SessionStatus.RUNNING.value,
            )
            await session.execute(stmt)
            await session.commit()
            return

        # Portable fallback: SELECT-then-INSERT/UPDATE. Two round trips.
        existing = (
            await session.execute(
                select(SessionRecord).where(
                    and_(
                        SessionRecord.session_id == session_id,
                        SessionRecord.user_id == user_id,
                    )
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            session.add(SessionRecord(**values))
        elif existing.status == SessionStatus.RUNNING.value:
            existing.last_activity_at = now
            if existing.dataset_id is None and dataset_id is not None:
                existing.dataset_id = dataset_id
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
    """Atomically add usage counters to the session row + per-model row.

    Only mutates sessions in ``running`` state — terminal sessions are
    frozen. Per-model accumulation runs against
    ``session_model_usage`` via an upsert so mixed-model sessions
    attribute correctly in ``cost-by-model``.
    """
    if tokens_in == 0 and tokens_out == 0 and cost_usd == 0.0 and not errored and model is None:
        return

    engine = get_relational_engine()

    async with engine.get_async_session() as session:
        # 1) Session-level aggregate. Gated on running status so a
        #    terminal session doesn't accrue straggler charges.
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

        if values:
            await session.execute(
                update(SessionRecord)
                .where(
                    and_(
                        SessionRecord.session_id == session_id,
                        SessionRecord.user_id == user_id,
                        SessionRecord.status == SessionStatus.RUNNING.value,
                    )
                )
                .values(**values)
            )

        # 2) Per-model row — only when there's usage to credit.
        if model and (tokens_in or tokens_out or cost_usd):
            now = datetime.now(timezone.utc)
            bind = await session.connection()
            dialect = _dialect_name(bind)

            if dialect in ("sqlite", "postgresql"):
                insert = sqlite_insert if dialect == "sqlite" else pg_insert
                stmt = insert(SessionModelUsage).values(
                    session_id=session_id,
                    user_id=user_id,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_usd,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["session_id", "user_id", "model"],
                    set_={
                        "tokens_in": SessionModelUsage.tokens_in + tokens_in,
                        "tokens_out": SessionModelUsage.tokens_out + tokens_out,
                        "cost_usd": SessionModelUsage.cost_usd + cost_usd,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
            else:
                # Portable fallback: SELECT then INSERT/UPDATE.
                existing = (
                    await session.execute(
                        select(SessionModelUsage).where(
                            and_(
                                SessionModelUsage.session_id == session_id,
                                SessionModelUsage.user_id == user_id,
                                SessionModelUsage.model == model,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        SessionModelUsage(
                            session_id=session_id,
                            user_id=user_id,
                            model=model,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            cost_usd=cost_usd,
                            updated_at=now,
                        )
                    )
                else:
                    existing.tokens_in = existing.tokens_in + tokens_in
                    existing.tokens_out = existing.tokens_out + tokens_out
                    existing.cost_usd = existing.cost_usd + cost_usd
                    existing.updated_at = now

        await session.commit()


async def mark_ended(
    *,
    session_id: str,
    user_id: UUIDType,
    status: SessionStatus,
) -> None:
    """Transition to a terminal status (completed / failed)."""
    if status == SessionStatus.RUNNING or status == SessionStatus.ABANDONED:
        raise ValueError(f"mark_ended requires a terminal status (completed/failed), got {status}")

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


async def delete_session_lifecycle(
    *,
    session_id: str,
    user_id: str | UUIDType,
) -> bool:
    """Delete a session's lifecycle row and its per-model usage rows.

    Best-effort companion to SessionManager.delete_session: a malformed
    user_id or a relational failure returns False without breaking an
    otherwise successful cache deletion.
    """
    try:
        user_uuid = UUIDType(str(user_id))
    except (ValueError, TypeError):
        return False

    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            await session.execute(
                delete(SessionModelUsage).where(
                    and_(
                        SessionModelUsage.session_id == session_id,
                        SessionModelUsage.user_id == user_uuid,
                    )
                )
            )
            result = await session.execute(
                delete(SessionRecord).where(
                    and_(
                        SessionRecord.session_id == session_id,
                        SessionRecord.user_id == user_uuid,
                    )
                )
            )
            await session.commit()
            return bool(result.rowcount)
    except Exception as exc:
        logger.warning("Failed to delete lifecycle rows for session %s: %s", session_id, exc)
        return False


async def get_session_dataset(
    *,
    session_id: str,
    user_id: str | UUIDType,
) -> Optional[tuple[UUIDType, UUIDType]]:
    """Return (dataset_id, dataset_owner_id) for a session's attributed dataset.

    The dataset owner identifies the database context that session vectors were
    written under. Best-effort: returns None when the session has no dataset
    attribution, the dataset row is gone, or the lookup fails.
    """
    try:
        user_uuid = UUIDType(str(user_id))
    except (ValueError, TypeError):
        return None

    try:
        from cognee.modules.data.models import Dataset

        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            row = (
                await session.execute(
                    select(Dataset.id, Dataset.owner_id)
                    .join(SessionRecord, SessionRecord.dataset_id == Dataset.id)
                    .where(
                        and_(
                            SessionRecord.session_id == session_id,
                            SessionRecord.user_id == user_uuid,
                        )
                    )
                )
            ).first()
        if row is None or row.owner_id is None:
            return None
        return row.id, row.owner_id
    except Exception as exc:
        logger.debug("Failed to resolve dataset for session %s: %s", session_id, exc)
        return None


_binding_lookup_failed = False


async def check_session_dataset_binding(
    *,
    session_id: str,
    user_id: str | UUIDType,
    dataset_id: str | UUIDType | None,
) -> None:
    """Raise ``SessionDatasetMismatchError`` when a write targets the wrong dataset.

    Sessions live in exactly one dataset: the first write binds the session
    (``ensure_and_touch_session`` fills ``dataset_id`` once) and every later
    write must target the same dataset. No-ops when the proposed dataset is
    unknown, the session has no binding yet, or the binding lookup fails —
    the session_records table is best-effort infrastructure, so only a
    genuine mismatch raises.

    Note: check-then-write is not atomic — two concurrent *first* writes to
    one session can both pass before either binds. Real callers derive one
    session per conversation with a fixed dataset, so this is accepted; an
    atomic claim in ``ensure_and_touch_session`` is the fix if that changes.
    """
    global _binding_lookup_failed
    from cognee.modules.session_lifecycle.exceptions import SessionDatasetMismatchError

    if dataset_id is None or not session_id:
        return
    try:
        user_uuid = UUIDType(str(user_id))
        dataset_uuid = UUIDType(str(dataset_id))
    except (ValueError, TypeError):
        return

    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            bound = (
                await session.execute(
                    select(SessionRecord.dataset_id).where(
                        and_(
                            SessionRecord.session_id == session_id,
                            SessionRecord.user_id == user_uuid,
                        )
                    )
                )
            ).scalar_one_or_none()
    except Exception as exc:
        if not _binding_lookup_failed:
            _binding_lookup_failed = True
            logger.warning(
                "Session binding lookup failed (%s); one-dataset-per-session enforcement "
                "is skipped while lookups fail. Subsequent failures log at debug.",
                exc,
            )
        else:
            logger.debug(
                "Session binding lookup failed for %s (%s); skipping enforcement", session_id, exc
            )
        return

    if bound is not None and bound != dataset_uuid:
        raise SessionDatasetMismatchError(session_id, bound, dataset_uuid)


async def delete_sessions_for_dataset(dataset_id: UUIDType) -> None:
    """Delete every session bound to a dataset that is being deleted.

    Sessions live in exactly one dataset and their content quotes its
    documents, so they share the dataset's blast radius — including sessions
    that *other* users attributed to a shared dataset. Best-effort: dataset
    deletion must not fail on the optional session infrastructure.
    """
    try:
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            rows = (
                await session.execute(
                    select(SessionRecord.session_id, SessionRecord.user_id).where(
                        SessionRecord.dataset_id == dataset_id
                    )
                )
            ).all()
    except Exception as exc:
        logger.warning("Failed to list sessions of deleted dataset %s: %s", dataset_id, exc)
        return

    from cognee.infrastructure.session.get_session_manager import get_session_manager

    manager = get_session_manager()
    for session_id, user_id in rows:
        try:
            # Cache content + vectors + lifecycle rows when the cache is available;
            # the explicit lifecycle delete below covers cache-less deployments
            # (delete_session no-ops without a cache engine, and it is idempotent).
            await manager.delete_session(user_id=str(user_id), session_id=session_id)
            await delete_session_lifecycle(session_id=session_id, user_id=user_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete session %s of deleted dataset %s: %s",
                session_id,
                dataset_id,
                exc,
            )


def get_effective_status_sql():
    """Return a SQL expression that evaluates to the effective status.

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
    user_ids: Optional[list[UUIDType]] = None,
    permitted_dataset_ids: Optional[list[UUIDType]] = None,
    prefer_other_owner: bool = False,
) -> Optional[SessionRecord]:
    """Fetch a session row visible to the caller.

    Returns the row if the caller (or their child agents, via
    ``user_ids``) owns the session OR if the session's dataset is in
    ``permitted_dataset_ids``. Returns None otherwise.

    The same ``session_id`` can exist under multiple owners (it's only
    unique per user in the composite PK). When the query matches
    multiple rows and ``prefer_other_owner`` is True, returns one
    whose owner is NOT the caller (useful for cache-reads via dataset
    grants). Otherwise returns the first match.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        if user_ids is not None and len(user_ids) > 0:
            visibility_terms = [SessionRecord.user_id.in_(user_ids)]
        else:
            visibility_terms = [SessionRecord.user_id == user_id]
        if permitted_dataset_ids:
            visibility_terms.append(SessionRecord.dataset_id.in_(permitted_dataset_ids))
        result = await session.execute(
            select(SessionRecord).where(
                and_(
                    SessionRecord.session_id == session_id,
                    or_(*visibility_terms) if len(visibility_terms) > 1 else visibility_terms[0],
                )
            )
        )
        rows = list(result.scalars().all())
        if not rows:
            return None
        if prefer_other_owner:
            non_owner = [r for r in rows if r.user_id != user_id]
            if non_owner:
                return non_owner[0]
        return rows[0]


@dataclass(slots=True)
class SessionRowWithStatus:
    """List envelope for ``list_session_rows`` — the SessionRecord plus
    the computed effective status, without attaching dynamic
    attributes to the ORM instance."""

    record: SessionRecord
    effective_status: str

    def to_dict(self) -> dict:
        d = self.record.to_dict()
        d["effective_status"] = self.effective_status
        return d


@dataclass(slots=True)
class SessionListPage:
    """Paginated list envelope."""

    sessions: list[SessionRowWithStatus]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.sessions) < self.total


async def list_session_rows(
    *,
    user_id: Optional[UUIDType] = None,
    user_ids: Optional[list[UUIDType]] = None,
    permitted_dataset_ids: Optional[list[UUIDType]] = None,
    since: Optional[datetime] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    order_by: str = "last_activity_at",
    descending: bool = True,
) -> SessionListPage:
    """List sessions with pagination metadata.

    Visibility: returns sessions the caller owns (or their child
    agents own, via ``user_ids``) OR sessions whose ``dataset_id``
    is in ``permitted_dataset_ids`` (read permission granted at the
    dataset level).

    status_filter accepts the effective status (including
    ``abandoned``) — the SQL predicate handles the abandoned-by-time
    inference.
    """
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        eff = get_effective_status_sql()

        # Ownership / permission predicate.
        visibility_terms = []
        if user_ids is not None and len(user_ids) > 0:
            visibility_terms.append(SessionRecord.user_id.in_(user_ids))
        elif user_id is not None:
            visibility_terms.append(SessionRecord.user_id == user_id)
        if permitted_dataset_ids:
            visibility_terms.append(SessionRecord.dataset_id.in_(permitted_dataset_ids))

        filters = []
        if visibility_terms:
            filters.append(
                or_(*visibility_terms) if len(visibility_terms) > 1 else visibility_terms[0]
            )
        if since is not None:
            filters.append(SessionRecord.last_activity_at >= since)
        if status_filter:
            filters.append(eff == status_filter)

        # Count before pagination so the caller can render
        # "showing N of M".
        count_stmt = select(func.count()).select_from(SessionRecord)
        if filters:
            count_stmt = count_stmt.where(and_(*filters))
        total = (await session.execute(count_stmt)).scalar_one()

        sortable = {
            "last_activity_at": SessionRecord.last_activity_at,
            "started_at": SessionRecord.started_at,
            "ended_at": SessionRecord.ended_at,
            "cost_usd": SessionRecord.cost_usd,
            "tokens_in": SessionRecord.tokens_in,
            "tokens_out": SessionRecord.tokens_out,
        }
        sort_col = sortable.get(order_by, SessionRecord.last_activity_at)

        rows_stmt = select(SessionRecord, eff.label("effective_status"))
        if filters:
            rows_stmt = rows_stmt.where(and_(*filters))
        rows_stmt = rows_stmt.order_by(sort_col.desc() if descending else sort_col.asc())
        rows_stmt = rows_stmt.limit(limit).offset(offset)

        rows = (await session.execute(rows_stmt)).all()

        return SessionListPage(
            sessions=[SessionRowWithStatus(record=r[0], effective_status=r[1]) for r in rows],
            total=int(total),
            limit=limit,
            offset=offset,
        )


# Backward-compatibility shims ------------------------------------------------


async def ensure_session(*, session_id, user_id, dataset_id=None):
    """Deprecated: prefer ``ensure_and_touch_session``. Kept for callers
    that only want the "row must exist" half."""
    await ensure_and_touch_session(session_id=session_id, user_id=user_id, dataset_id=dataset_id)


async def touch_session(*, session_id, user_id, dataset_id=None):
    """Deprecated: prefer ``ensure_and_touch_session``."""
    await ensure_and_touch_session(session_id=session_id, user_id=user_id, dataset_id=dataset_id)


_session_record_write_failed = False


async def record_session_activity(
    user_id: str,
    session_id: str,
    *,
    dataset_id: str | UUIDType | None = None,
    errored: bool = False,
) -> None:
    """Write a lifecycle heartbeat for a session: upsert + touch the SessionRecord row.

    Accepts a string ``user_id`` (coerced to UUID). ``dataset_id`` fills the
    row's dataset attribution when it is not set yet. Swallows failures — the
    session_records table is optional for SessionManager correctness — but logs once at
    WARNING per process so silent breakage stays visible in ops.
    """
    global _session_record_write_failed

    try:
        try:
            user_uuid = UUIDType(str(user_id))
        except (ValueError, TypeError):
            return

        try:
            dataset_uuid = UUIDType(str(dataset_id)) if dataset_id is not None else None
        except (ValueError, TypeError):
            dataset_uuid = None

        await ensure_and_touch_session(
            session_id=session_id, user_id=user_uuid, dataset_id=dataset_uuid
        )
        if errored:
            await accumulate_usage(session_id=session_id, user_id=user_uuid, errored=True)
    except Exception as exc:
        if not _session_record_write_failed:
            _session_record_write_failed = True
            logger.warning(
                "session_records write failed (%s); subsequent failures will log at debug. "
                "Check alembic migrations for the session_records table.",
                exc,
            )
        else:
            logger.debug("session_records write failed (%s)", exc)
