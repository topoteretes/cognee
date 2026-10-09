"""Per-session lock primitives — in-process asyncio registry.

Three primitives:

* ``session_lock(session_id, op)`` — async context manager that
  serializes concurrent tasks on the same ``(session_id, op)`` key.
  Used for short read-modify-write flows (``update_qa``,
  ``add_feedback``, ``delete_qa``).

* ``session_turn_lock(user_id, session_id)`` — async context manager
  that serializes whole session turns for one cache identity, so two
  quick turns cannot read the same state and overwrite each other.

* ``acquire_improve_lock_many(keys)`` / ``release_improve_lock_many(keys)``
  — the claim a long-running ``improve()`` holds. Like the per-dataset
  lock pipeline runs take, it waits: overlapping improves queue one after
  another instead of skipping.

Scope: single-worker FastAPI. For multi-worker deployments, layer a
row-level SQL advisory lock or Redis SETNX on top — the call sites
are factored so that's a local change.
"""

import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from typing import Any

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


# ----- Improve claim ---------------------------------------------------------
#
# One asyncio.Lock per claim key. A run takes all of its keys in sorted order,
# so two runs that share any key queue one after the other and can never hold
# one key each while waiting for the other's. Registered per event loop for the
# same reason as the turn locks above.
_improve_lock_registries: dict[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = {}


def _improve_locks(keys: Iterable[str]) -> list[asyncio.Lock]:
    registry = _improve_lock_registries.setdefault(asyncio.get_running_loop(), {})
    return [registry.setdefault(key, asyncio.Lock()) for key in sorted({k for k in keys if k})]


async def acquire_improve_lock_many(keys: Iterable[str]) -> None:
    """Wait until every key in ``keys`` is free, then hold them all.

    One claim per ``improve()`` run, keyed by what the run touches (see
    ``improve_lock_keys``). The caller MUST release the same keys exactly once
    with ``release_improve_lock_many``. If the wait is cancelled, the keys
    already taken are released before the cancellation propagates.
    """
    acquired: list[asyncio.Lock] = []
    try:
        for lock in _improve_locks(keys):
            await lock.acquire()
            acquired.append(lock)
    except BaseException:
        for lock in reversed(acquired):
            lock.release()
        raise


async def release_improve_lock_many(keys: Iterable[str]) -> None:
    """Release every key taken by ``acquire_improve_lock_many``.

    Releasing a key that is not held raises: a double release would hand the
    next queued run a claim it shares with a run still in progress.
    """
    for lock in reversed(_improve_locks(keys)):
        lock.release()


def improve_lock_keys(
    session_ids: Iterable[str] | None, dataset_id: Any, user_id: Any
) -> tuple[str, ...]:
    """The claim keys for one improve run: its session ids plus its dataset id.

    Session keys carry the user id because session state is scoped per
    ``(user_id, session_id)`` everywhere else — two users who both call a
    session "chat" must never block each other. Every run also claims the
    dataset key, session-fed or not, so a session-keyed bridge run and a
    dataset-keyed run over the same dataset exclude each other — improves for
    one dataset run one after another.
    """
    sessions = tuple(
        f"session:{user_id}:{session_id}" for session_id in (session_ids or ()) if session_id
    )
    return (*sessions, f"dataset:{dataset_id}")
