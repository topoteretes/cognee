"""Per-session lock primitives — in-process asyncio registry.

Three primitives:

* ``session_lock(session_id, op)`` — async context manager that
  serializes concurrent tasks on the same ``(session_id, op)`` key.
  Used for short read-modify-write flows (``update_qa``,
  ``add_feedback``, ``delete_qa``).

* ``session_turn_lock(user_id, session_id)`` — async context manager
  that serializes whole session turns for one cache identity, so two
  quick turns cannot read the same state and overwrite each other.

* ``try_acquire_improve_lock(session_id, user_id)`` /
  ``release_improve_lock(session_id, user_id, token)`` — non-blocking
  claim for long-running ``improve()`` calls, backed by the
  ``session_records`` row so it holds across workers (SDK-593). The
  claim is one conditional UPDATE, so two callers can't both see
  "free" and both think they won. A claim older than
  ``IMPROVE_LOCK_TTL_SECONDS`` counts as expired and may be taken
  over, so a hung or killed holder can never wedge a session for good.
  ``request_improve_rerun`` carries the busy caller's "there is a newer
  tail" signal to the holder, whose release consumes it.

The first two primitives are process-local (asyncio); only the improve
lock crosses workers.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import UUID as UUIDType

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_lock")


_locks: dict[tuple[str, str], asyncio.Lock] = {}
_registry_lock = asyncio.Lock()


async def _get_lock(session_id: str, op: str) -> asyncio.Lock:
    key = (session_id, op)
    async with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _locks[key] = lock
        return lock


@asynccontextmanager
async def session_lock(session_id: str, op: str = "write") -> AsyncGenerator[None, None]:
    """Serialize concurrent operations on the same session/op pair.

    Usage::

        async with session_lock(session_id, "update_qa"):
            ...  # read-modify-write
    """
    if not session_id:
        yield
        return

    lock = await _get_lock(session_id, op)
    async with lock:
        yield


# Turn locks are registered per event loop, unlike ``_locks`` above: an ``asyncio.Lock``
# is bound to the loop that awaited it, so reusing one from a different loop raises
# RuntimeError. Keying by loop means a fresh loop always gets its own empty table, so a
# lock is never reused across loops. Like ``_locks``, entries are never expired — that's
# the same trade-off the registry above already makes.
_turn_lock_registries: dict[asyncio.AbstractEventLoop, dict[tuple[str, str], asyncio.Lock]] = {}
_turn_registry_guard = asyncio.Lock()


async def _get_turn_lock(user_id: Any, session_id: Any) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = (str(user_id), str(session_id))
    async with _turn_registry_guard:
        registry = _turn_lock_registries.setdefault(loop, {})
        lock = registry.get(key)
        if lock is None:
            lock = asyncio.Lock()
            registry[key] = lock
        return lock


@asynccontextmanager
async def session_turn_lock(user_id: Any, session_id: Any) -> AsyncGenerator[None, None]:
    """Serialize full concurrent turns for one cache user/session identity."""
    if not user_id or not session_id:
        yield
        return

    lock = await _get_turn_lock(user_id, session_id)
    async with lock:
        yield


# ----- Cross-worker improve-lock claim ------------------------------------------
#
# The lock lives on the ``session_records`` row (``improve_lock_token`` /
# ``improve_lock_acquired_at`` / ``improve_rerun_requested``). Every mutation
# is a single conditional UPDATE whose WHERE clause names the state the caller
# observed, so the row's own atomicity decides the race — on SQLite and on
# Postgres alike, across any number of workers. An earlier in-process set
# excluded nothing across workers and had no TTL (SDK-593).

DEFAULT_IMPROVE_LOCK_TTL_SECONDS = 1800


def improve_lock_ttl_seconds() -> int:
    """TTL after which a held improve lock counts as expired (``IMPROVE_LOCK_TTL_SECONDS``)."""
    raw = os.environ.get("IMPROVE_LOCK_TTL_SECONDS", "")
    try:
        value = int(raw) if raw else DEFAULT_IMPROVE_LOCK_TTL_SECONDS
    except ValueError:
        return DEFAULT_IMPROVE_LOCK_TTL_SECONDS
    return value if value > 0 else DEFAULT_IMPROVE_LOCK_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class ImproveLockStatus:
    """What a caller that did NOT get the lock learns about the holder."""

    busy: bool
    holder_age_seconds: float | None = None
    rerun_requested: bool = False


class ImproveLockRelease(str, Enum):
    """Outcome of ``release_improve_lock``."""

    RELEASED = "released"  # we let go; the lock is free
    RERUN = "rerun"  # a rerun request was pending: consumed, and we STILL hold the lock
    LOST = "lost"  # the lock is no longer ours (TTL takeover); nothing to release


def _coerce_user_id(user_id: Any) -> UUIDType:
    return user_id if isinstance(user_id, UUIDType) else UUIDType(str(user_id))


def _rowcount(result: Any) -> int:
    """Rows matched by a DML statement (0 when the driver does not report it)."""
    count = getattr(result, "rowcount", 0)
    return int(count) if isinstance(count, int) and count > 0 else 0


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _lock_age_seconds(acquired_at: datetime | None, now: datetime) -> float | None:
    acquired_at = _as_utc(acquired_at)
    if acquired_at is None:
        return None
    return max(0.0, (now - acquired_at).total_seconds())


async def _ensure_session_row(session_id: str, user_id: UUIDType) -> None:
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session

    await ensure_and_touch_session(session_id=session_id, user_id=user_id)


async def try_acquire_improve_lock(session_id: str, user_id: Any) -> str | None:
    """Atomically claim the improve-lock for ``(user_id, session_id)``.

    Returns the holder token iff we got it; the caller MUST pass that token to
    ``release_improve_lock`` when done (use try/finally). Returns ``None``
    immediately when another run holds a live lock — callers should no-op (and
    ``request_improve_rerun``) rather than wait. A lock older than the TTL is
    taken over with a warning.
    """
    if not session_id or not user_id:
        return uuid.uuid4().hex  # no-op sessions don't need exclusion

    from sqlalchemy import and_, select, update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    user_uuid = _coerce_user_id(user_id)
    token = uuid.uuid4().hex
    engine = get_relational_engine()

    for attempt in range(2):
        now = datetime.now(timezone.utc)
        async with engine.get_async_session() as session:
            row = (
                await session.execute(
                    select(
                        SessionRecord.improve_lock_token,
                        SessionRecord.improve_lock_acquired_at,
                    ).where(
                        and_(
                            SessionRecord.session_id == session_id,
                            SessionRecord.user_id == user_uuid,
                        )
                    )
                )
            ).first()

            if row is None:
                if attempt == 0:
                    # SDK-only sessions never went through remember(), so the
                    # lifecycle row does not exist yet. Create it and retry.
                    await _ensure_session_row(session_id, user_uuid)
                    continue
                logger.warning(
                    "improve lock: session row for '%s' could not be created; "
                    "proceeding without cross-worker exclusion",
                    session_id,
                )
                return token

            held_token, acquired_at = row.improve_lock_token, row.improve_lock_acquired_at
            age = _lock_age_seconds(acquired_at, now)
            expired = held_token is not None and (age is None or age > improve_lock_ttl_seconds())
            if held_token is not None and not expired:
                return None

            # Claim only if the row still looks the way we just observed it.
            observed = (
                SessionRecord.improve_lock_token.is_(None)
                if held_token is None
                else SessionRecord.improve_lock_token == held_token
            )
            result = await session.execute(
                update(SessionRecord)
                .where(
                    and_(
                        SessionRecord.session_id == session_id,
                        SessionRecord.user_id == user_uuid,
                        observed,
                    )
                )
                .values(
                    improve_lock_token=token,
                    improve_lock_acquired_at=now,
                    # A fresh holder starts a full watermark pass, which is
                    # exactly what a pending rerun request asked for.
                    improve_rerun_requested=False,
                )
            )
            await session.commit()
            if _rowcount(result) != 1:
                return None  # lost the race to a concurrent claimant
            if expired:
                logger.warning(
                    "improve lock: took over an expired lock on session '%s' "
                    "(held for %s s, TTL %d s)",
                    session_id,
                    "unknown" if age is None else int(age),
                    improve_lock_ttl_seconds(),
                )
            return token
    return None


async def release_improve_lock(
    session_id: str | None, user_id: Any, token: str | None, *, force: bool = False
) -> ImproveLockRelease:
    """Release the improve-lock if ``token`` still holds it and no rerun is pending.

    Three conditional UPDATEs, each atomic on its own, no read of a snapshot:

    1. clear the token where it is ours AND no rerun is pending -> ``RELEASED``;
    2. otherwise clear the rerun flag where the token is ours AND the flag is
       set -> ``RERUN``: a busy caller was promised its newer tail would be
       covered, so the request is consumed and the caller must run one more pass
       while still holding the lock;
    3. otherwise the lock is not ours (taken over after the TTL) -> ``LOST``.

    Because the flag is checked and the token cleared in the same statement, a
    request cannot land between "checked" and "released" and be dropped. And
    because step 2 also names our token, a takeover between steps 1 and 2 makes
    it match nothing, so we never consume another holder's request.

    ``force=True`` clears the token regardless of the flag (error paths, the
    pass-count bound). A pending flag is deliberately left in place: the next
    acquirer clears it by starting a full watermark pass, which is exactly what
    the request asked for.
    """
    if not session_id or not user_id or not token:
        return ImproveLockRelease.RELEASED

    from sqlalchemy import and_, update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    ours = and_(
        SessionRecord.session_id == session_id,
        SessionRecord.user_id == _coerce_user_id(user_id),
        SessionRecord.improve_lock_token == token,
    )
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        release_where = (
            ours if force else and_(ours, SessionRecord.improve_rerun_requested.is_(False))
        )
        released = await session.execute(
            update(SessionRecord)
            .where(release_where)
            .values(improve_lock_token=None, improve_lock_acquired_at=None)
        )
        await session.commit()
        if _rowcount(released) == 1:
            return ImproveLockRelease.RELEASED
        if force:
            return ImproveLockRelease.LOST

        consumed = await session.execute(
            update(SessionRecord)
            .where(and_(ours, SessionRecord.improve_rerun_requested.is_(True)))
            .values(improve_rerun_requested=False)
        )
        await session.commit()
        if _rowcount(consumed) == 1:
            return ImproveLockRelease.RERUN
        return ImproveLockRelease.LOST


async def request_improve_rerun(session_id: str, user_id: Any) -> ImproveLockStatus:
    """Ask the current holder for one more pass; report the holder's age.

    Called by a run that found the lock busy. The flag is only set while the
    lock is actually held, so a caller can never leave a stale request behind
    on a free session. Returns ``busy=False`` when the holder released in the
    meantime — the caller may then simply try to acquire again.
    """
    if not session_id or not user_id:
        return ImproveLockStatus(busy=False)

    from sqlalchemy import and_, select, update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.session_lifecycle.models import SessionRecord

    user_uuid = _coerce_user_id(user_id)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        result = await session.execute(
            update(SessionRecord)
            .where(
                and_(
                    SessionRecord.session_id == session_id,
                    SessionRecord.user_id == user_uuid,
                    SessionRecord.improve_lock_token.is_not(None),
                )
            )
            .values(improve_rerun_requested=True)
        )
        await session.commit()
        if _rowcount(result) != 1:
            return ImproveLockStatus(busy=False)

        acquired_at = (
            await session.execute(
                select(SessionRecord.improve_lock_acquired_at).where(
                    and_(
                        SessionRecord.session_id == session_id,
                        SessionRecord.user_id == user_uuid,
                    )
                )
            )
        ).scalar_one_or_none()

    return ImproveLockStatus(
        busy=True,
        holder_age_seconds=_lock_age_seconds(acquired_at, datetime.now(timezone.utc)),
        rerun_requested=True,
    )
