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

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Sequence
from uuid import UUID as UUIDType

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from .models import OperationModelUsage, OperationRecord, SessionModelUsage, SessionRecord

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


async def _ensure_and_touch_row(
    *,
    record_cls,
    id_field: str,
    id_value: str,
    user_id: UUIDType,
    dataset_id: Optional[UUIDType],
    extra_values: dict,
) -> None:
    """Upsert a lifecycle row (session or operation) in one round trip.

    Creates the row if absent (status=running). If present AND still
    running, bumps ``last_activity_at``. Terminal rows are left untouched
    so a late straggler can't accidentally resurrect them. Also fills in
    ``dataset_id`` when currently null. Shared by ``ensure_and_touch_session``
    and ``ensure_and_touch_operation``.
    """
    now = datetime.now(timezone.utc)
    engine = get_relational_engine()
    id_col = getattr(record_cls, id_field)

    async with engine.get_async_session() as session:
        bind = await session.connection()
        dialect = _dialect_name(bind)

        values = {
            id_field: id_value,
            "user_id": user_id,
            "dataset_id": dataset_id,
            "status": SessionStatus.RUNNING.value,
            "started_at": now,
            "last_activity_at": now,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            **extra_values,
        }

        if dialect in ("sqlite", "postgresql"):
            insert = sqlite_insert if dialect == "sqlite" else pg_insert
            stmt = insert(record_cls).values(**values)
            set_ = {"last_activity_at": now}
            # Back-fill a previously-unset dataset_id.
            set_["dataset_id"] = case(
                (record_cls.dataset_id.is_(None), dataset_id),
                else_=record_cls.dataset_id,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[id_field, "user_id"],
                set_=set_,
                where=record_cls.status == SessionStatus.RUNNING.value,
            )
            await session.execute(stmt)
            await session.commit()
            return

        # Portable fallback: SELECT-then-INSERT/UPDATE. Two round trips.
        existing = (
            await session.execute(
                select(record_cls).where(and_(id_col == id_value, record_cls.user_id == user_id))
            )
        ).scalar_one_or_none()
        if existing is None:
            session.add(record_cls(**values))
        elif existing.status == SessionStatus.RUNNING.value:
            existing.last_activity_at = now
            if existing.dataset_id is None and dataset_id is not None:
                existing.dataset_id = dataset_id
        await session.commit()


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
    await _ensure_and_touch_row(
        record_cls=SessionRecord,
        id_field="session_id",
        id_value=session_id,
        user_id=user_id,
        dataset_id=dataset_id,
        extra_values={"error_count": 0},
    )


async def _accumulate_usage_row(
    *,
    record_cls,
    model_usage_cls,
    id_field: str,
    id_value: str,
    user_id: UUIDType,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    model: Optional[str] = None,
    errored: bool = False,
    touch_activity_on_model: bool = False,
) -> None:
    """Atomically add usage counters to a lifecycle row + its per-model row.

    Only mutates rows in ``running`` state — terminal rows are frozen.
    Per-model accumulation runs against ``model_usage_cls`` via an upsert so
    mixed-model runs attribute correctly in ``cost-by-model``. Shared by
    ``accumulate_usage`` and ``accumulate_operation_usage``.

    ``errored`` bumps ``error_count`` — only ``SessionRecord`` has that
    column, so operations must pass ``errored=False`` (the default).
    ``touch_activity_on_model`` additionally bumps ``last_activity_at``
    when a model is credited (``accumulate_operation_usage``'s behavior).
    """
    engine = get_relational_engine()
    id_col = getattr(record_cls, id_field)
    model_id_col = getattr(model_usage_cls, id_field)

    async with engine.get_async_session() as session:
        # 1) Row-level aggregate. Gated on running status so a terminal
        #    row doesn't accrue straggler charges.
        values = {}
        if tokens_in:
            values["tokens_in"] = record_cls.tokens_in + tokens_in
        if tokens_out:
            values["tokens_out"] = record_cls.tokens_out + tokens_out
        if cost_usd:
            values["cost_usd"] = record_cls.cost_usd + cost_usd
        if errored:
            values["error_count"] = record_cls.error_count + 1
        if model:
            values["last_model"] = model
            if touch_activity_on_model:
                values["last_activity_at"] = datetime.now(timezone.utc)

        if values:
            await session.execute(
                update(record_cls)
                .where(
                    and_(
                        id_col == id_value,
                        record_cls.user_id == user_id,
                        record_cls.status == SessionStatus.RUNNING.value,
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
                stmt = insert(model_usage_cls).values(
                    **{id_field: id_value},
                    user_id=user_id,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_usd,
                    updated_at=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[id_field, "user_id", "model"],
                    set_={
                        "tokens_in": model_usage_cls.tokens_in + tokens_in,
                        "tokens_out": model_usage_cls.tokens_out + tokens_out,
                        "cost_usd": model_usage_cls.cost_usd + cost_usd,
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
            else:
                # Portable fallback: SELECT then INSERT/UPDATE.
                existing = (
                    await session.execute(
                        select(model_usage_cls).where(
                            and_(
                                model_id_col == id_value,
                                model_usage_cls.user_id == user_id,
                                model_usage_cls.model == model,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        model_usage_cls(
                            **{id_field: id_value},
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

    await _accumulate_usage_row(
        record_cls=SessionRecord,
        model_usage_cls=SessionModelUsage,
        id_field="session_id",
        id_value=session_id,
        user_id=user_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        model=model,
        errored=errored,
    )


async def _mark_ended_row(
    *,
    record_cls,
    id_field: str,
    id_value: str,
    user_id: UUIDType,
    status: SessionStatus,
    label: str,
) -> None:
    """Transition a lifecycle row to a terminal status (completed / failed).

    Shared by ``mark_ended`` and ``mark_operation_ended``.
    """
    if status == SessionStatus.RUNNING or status == SessionStatus.ABANDONED:
        raise ValueError(f"{label} requires a terminal status (completed/failed), got {status}")

    now = datetime.now(timezone.utc)
    engine = get_relational_engine()
    id_col = getattr(record_cls, id_field)
    async with engine.get_async_session() as session:
        stmt = (
            update(record_cls)
            .where(and_(id_col == id_value, record_cls.user_id == user_id))
            .values(status=status.value, ended_at=now)
        )
        await session.execute(stmt)
        await session.commit()


async def mark_ended(
    *,
    session_id: str,
    user_id: UUIDType,
    status: SessionStatus,
) -> None:
    """Transition to a terminal status (completed / failed)."""
    await _mark_ended_row(
        record_cls=SessionRecord,
        id_field="session_id",
        id_value=session_id,
        user_id=user_id,
        status=status,
        label="mark_ended",
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


# Operation lifecycle functions -----------------------------------------------

# In-process locks serializing writes per (operation_id, user_id). Pipelines
# like cognify issue many LLM calls concurrently (asyncio.gather) within a
# single operation scope; without this, each completion opens its own DB
# session and they all race to write the same OperationRecord row, thrashing
# SQLite's single-writer lock / Postgres row locks. Popped in
# mark_operation_ended. (Background operations that finish successfully never
# call mark_operation_ended today, so their lock entry outlives the request —
# a small, bounded memory cost consistent with their row also living on
# indefinitely until read-time abandonment.)
_operation_write_locks: dict[tuple[str, UUIDType], asyncio.Lock] = {}


def _get_operation_lock(operation_id: str, user_id: UUIDType) -> asyncio.Lock:
    key = (operation_id, user_id)
    lock = _operation_write_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _operation_write_locks[key] = lock
    return lock


async def ensure_and_touch_operation(
    *,
    operation_id: str,
    user_id: UUIDType,
    operation_type: str,
    dataset_id: Optional[UUIDType] = None,
) -> None:
    """Upsert the operation row in one round trip.

    Creates the row if absent (status=running). If present AND still
    running, bumps ``last_activity_at``. Terminal operations are left
    untouched. Mirrors ``ensure_and_touch_session``.
    """
    await _ensure_and_touch_row(
        record_cls=OperationRecord,
        id_field="operation_id",
        id_value=operation_id,
        user_id=user_id,
        dataset_id=dataset_id,
        extra_values={"operation_type": operation_type},
    )


async def accumulate_operation_usage(
    *,
    operation_id: str,
    user_id: UUIDType,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    model: Optional[str] = None,
) -> None:
    """Atomically add usage counters to the operation row + per-model row.

    Only mutates operations in ``running`` state. Mirrors ``accumulate_usage``.
    Serialized per-operation via ``_get_operation_lock`` since concurrent LLM
    calls within one pipeline run would otherwise all write the same row at
    once.
    """
    if tokens_in == 0 and tokens_out == 0 and cost_usd == 0.0 and model is None:
        return

    async with _get_operation_lock(operation_id, user_id):
        await _accumulate_usage_row(
            record_cls=OperationRecord,
            model_usage_cls=OperationModelUsage,
            id_field="operation_id",
            id_value=operation_id,
            user_id=user_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            model=model,
            touch_activity_on_model=True,
        )


async def mark_operation_ended(
    *,
    operation_id: str,
    user_id: UUIDType,
    status: SessionStatus,
) -> None:
    """Transition an operation to a terminal status (completed / failed)."""
    await _mark_ended_row(
        record_cls=OperationRecord,
        id_field="operation_id",
        id_value=operation_id,
        user_id=user_id,
        status=status,
        label="mark_operation_ended",
    )
    _operation_write_locks.pop((operation_id, user_id), None)


# Backward-compatibility shims ------------------------------------------------


async def ensure_session(*, session_id, user_id, dataset_id=None):
    """Deprecated: prefer ``ensure_and_touch_session``. Kept for callers
    that only want the "row must exist" half."""
    await ensure_and_touch_session(session_id=session_id, user_id=user_id, dataset_id=dataset_id)


async def touch_session(*, session_id, user_id, dataset_id=None):
    """Deprecated: prefer ``ensure_and_touch_session``."""
    await ensure_and_touch_session(session_id=session_id, user_id=user_id, dataset_id=dataset_id)


_session_record_write_failed = False


async def record_session_activity(user_id: str, session_id: str, *, errored: bool = False) -> None:
    """Write a lifecycle heartbeat for a session: upsert + touch the SessionRecord row.

    Accepts a string ``user_id`` (coerced to UUID). Swallows failures — the
    session_records table is optional for SessionManager correctness — but logs once at
    WARNING per process so silent breakage stays visible in ops.
    """
    global _session_record_write_failed

    try:
        try:
            user_uuid = UUIDType(str(user_id))
        except (ValueError, TypeError):
            return

        await ensure_and_touch_session(session_id=session_id, user_id=user_uuid)
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
