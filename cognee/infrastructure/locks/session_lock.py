"""Per-session lock primitives — in-process, event-loop-agnostic registry.

Three primitives:

* ``session_lock(session_id, op)`` — async context manager that
  serializes concurrent tasks on the same ``(session_id, op)`` key.
  Used for short read-modify-write flows (``update_qa``,
  ``add_feedback``, ``delete_qa``).

* ``session_turn_lock(user_id, session_id)`` — async context manager
  that serializes whole session turns for one cache identity, so two
  quick turns cannot read the same state and overwrite each other.

* ``acquire_improve_lock_many(keys)`` / ``release_improve_lock_many(keys)``
  — the session claim a long-running ``improve()`` holds on top of the
  per-dataset lock. It waits: improves sharing a session queue one after
  another instead of skipping.

The registries hand out :class:`LoopAgnosticLock` objects and are guarded
by ``threading.Lock`` (held only for dict/set ops): cached locks outlive
any single event loop, and an ``asyncio.Lock`` binds to the first loop that
awaits it (see ``loop_agnostic_lock.py`` for the failure modes).

Scope: single-worker FastAPI. For multi-worker deployments, layer a
row-level SQL advisory lock or Redis SETNX on top — the call sites
are factored so that's a local change.
"""

import threading
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from typing import Any

from cognee.shared.logging_utils import get_logger

from .loop_agnostic_lock import LoopAgnosticLock

logger = get_logger("session_lock")


_locks: dict[tuple[str, str], LoopAgnosticLock] = {}
_registry_lock = threading.Lock()


async def _get_lock(session_id: str, op: str) -> LoopAgnosticLock:
    key = (session_id, op)
    with _registry_lock:
        lock = _locks.get(key)
        if lock is None:
            lock = LoopAgnosticLock()
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


# Turn locks used to be registered per event loop as a workaround for
# asyncio.Lock's loop binding, which meant two turns on different loops were
# NOT serialized. LoopAgnosticLock removes the constraint: one registry, and
# turns on the same identity serialize regardless of which loop runs them.
# Like ``_locks``, entries are never expired — the same trade-off the registry
# above already makes.
_turn_locks: dict[tuple[str, str], LoopAgnosticLock] = {}
_turn_registry_guard = threading.Lock()


async def _get_turn_lock(user_id: Any, session_id: Any) -> LoopAgnosticLock:
    key = (str(user_id), str(session_id))
    with _turn_registry_guard:
        lock = _turn_locks.get(key)
        if lock is None:
            lock = LoopAgnosticLock()
            _turn_locks[key] = lock
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
# One LoopAgnosticLock per claim key. A run takes all of its keys in sorted
# order, so two runs that share any key queue one after the other and can never
# hold one key each while waiting for the other's. One registry for every loop,
# like the turn locks above: runs on different loops queue on the same claim.
_improve_locks_registry: dict[str, LoopAgnosticLock] = {}
_improve_registry_guard = threading.Lock()


def _improve_locks(keys: Iterable[str]) -> list[LoopAgnosticLock]:
    with _improve_registry_guard:
        return [
            _improve_locks_registry.setdefault(key, LoopAgnosticLock())
            for key in sorted({k for k in keys if k})
        ]


async def acquire_improve_lock_many(keys: Iterable[str]) -> None:
    """Wait until every key in ``keys`` is free, then hold them all.

    One claim per ``improve()`` run, keyed by what the run touches (see
    ``improve_lock_keys``). The caller MUST release the same keys exactly once
    with ``release_improve_lock_many``. If the wait is cancelled, the keys
    already taken are released before the cancellation propagates.
    """
    acquired: list[LoopAgnosticLock] = []
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


def improve_lock_keys(session_ids: Iterable[str] | None, user_id: Any) -> tuple[str, ...]:
    """The session claim keys for one improve run.

    Session keys carry the user id because session state is scoped per
    ``(user_id, session_id)`` everywhere else — two users who both call a
    session "chat" must never block each other. The dataset is not a key:
    improve holds the per-dataset lock for its whole run, like a pipeline run.
    """
    return tuple(
        f"session:{user_id}:{session_id}" for session_id in (session_ids or ()) if session_id
    )
