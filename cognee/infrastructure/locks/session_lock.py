"""Per-session lock primitives — in-process, event-loop-agnostic registry.

Three primitives:

* ``session_lock(session_id, op)`` — async context manager that
  serializes concurrent tasks on the same ``(session_id, op)`` key.
  Used for short read-modify-write flows (``update_qa``,
  ``add_feedback``, ``delete_qa``).

* ``session_turn_lock(user_id, session_id)`` — async context manager
  that serializes whole session turns for one cache identity, so two
  quick turns cannot read the same state and overwrite each other.

* ``try_acquire_improve_lock_many(keys)`` /
  ``release_improve_lock_many(keys)`` — non-blocking claim for
  long-running ``improve()`` calls. The claim is atomic: a
  registry-wide lock protects a set of held keys, and
  the check-and-add happens inside that critical section so two
  callers can't both see "free" and both think they won.
  ``request_improve_rerun_many`` / ``release_or_rerun_improve_lock_many``
  carry a lock loser's "there is a newer tail" signal to the holder,
  which then runs one more pass before releasing (SDK-593).

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


# ----- Non-blocking improve-lock claim ---------------------------------------
#
# asyncio.Lock has no ``acquire_nowait`` on current Python, and the
# obvious ``if lock.locked(): ... await lock.acquire()`` pattern is
# racy — two coroutines can both observe "free" at the check before
# either reaches the acquire. Use a plain set guarded by a
# registry-wide lock instead: the check-and-add happens inside the
# registry lock's critical section, so the test is atomic.

_improving_sessions: set[str] = set()
# Keys whose current holder was asked for one more pass by a run that found
# them held (``request_improve_rerun_many``). Consumed by
# ``release_or_rerun_improve_lock_many``; a fresh claim clears them, since a
# new holder starts with a full watermark pass anyway.
_rerun_requested: set[str] = set()
_improve_registry_lock = threading.Lock()


async def try_acquire_improve_lock_many(keys: Iterable[str]) -> bool:
    """Atomically claim the improve-lock for every key in ``keys``, or none.

    One claim per ``improve()`` run, keyed by what the run touches: every
    session id it was given plus ``f"dataset:{dataset_id}"`` (see
    ``improve_lock_keys``). Same held-set, same registry lock as the single-key claim; the
    all-or-nothing check-and-add happens inside one critical section, so two
    overlapping runs that share any key cannot both win. Returns ``True`` iff
    every key was claimed — the caller MUST then call
    ``release_improve_lock_many`` with the same keys (use try/finally).
    Empty or all-falsy ``keys`` need no exclusion and return ``True``.
    """
    wanted = [key for key in keys if key]
    if not wanted:
        return True

    with _improve_registry_lock:
        if any(key in _improving_sessions for key in wanted):
            return False
        _improving_sessions.update(wanted)
        _rerun_requested.difference_update(wanted)
        return True


async def release_improve_lock_many(keys: Iterable[str]) -> None:
    """Release every key claimed by ``try_acquire_improve_lock_many``.

    Not holder-scoped: it drops the keys whoever holds them, so a run must
    release its claim exactly once — a second release after another run
    re-claimed the keys would drop THAT run's claim. Unconditional about rerun
    requests: one still pending on a key is left for the next claimant, whose
    full pass covers it.
    """
    wanted = [key for key in keys if key]
    if not wanted:
        return
    with _improve_registry_lock:
        _improving_sessions.difference_update(wanted)


async def request_improve_rerun_many(keys: Iterable[str]) -> bool:
    """Ask whoever holds any of ``keys`` to run one more pass before letting go.

    Called by a run that lost its lock claim. Returns ``True`` iff at least one
    key is currently held, i.e. the request reached a holder; a request on a
    free key would have nobody to fulfil it and is not recorded. The caller can
    then return without retrying: everything above the watermarks when the
    holder's extra pass runs — including this caller's newer tail — is covered.
    """
    wanted = [key for key in keys if key]
    if not wanted:
        return False
    with _improve_registry_lock:
        held = [key for key in wanted if key in _improving_sessions]
        if not held:
            return False
        _rerun_requested.update(held)
        return True


async def has_pending_improve_rerun(keys: Iterable[str]) -> bool:
    """Read-only: is a rerun request pending on any of ``keys``? Consumes nothing."""
    wanted = [key for key in keys if key]
    if not wanted:
        return False
    with _improve_registry_lock:
        return any(key in _rerun_requested for key in wanted)


async def release_or_rerun_improve_lock_many(
    keys: Iterable[str], *, rerun_keys: Iterable[str]
) -> bool:
    """Release every key — unless a rerun is pending on one of ``rerun_keys``.

    One critical section decides both: when a request is pending on a
    ``rerun_keys`` key we still hold, the request is consumed, EVERY key stays
    claimed, and ``False`` says "run the stages once more". Otherwise all
    ``keys`` are released and ``True`` is returned. Doing the check and the
    release under the same registry lock means no request can land between
    "checked" and "released" and be lost.
    """
    wanted = [key for key in keys if key]
    watched = [key for key in rerun_keys if key]
    with _improve_registry_lock:
        pending = [key for key in watched if key in _rerun_requested and key in _improving_sessions]
        if pending:
            _rerun_requested.difference_update(pending)
            return False
        _improving_sessions.difference_update(wanted)
        return True


def improve_lock_keys(
    session_ids: Iterable[str] | None, dataset_id: Any, user_id: Any
) -> tuple[str, ...]:
    """The claim keys for one improve run: its session ids plus its dataset id.

    Session keys carry the user id because session state is scoped per
    ``(user_id, session_id)`` everywhere else — two users who both call a
    session "chat" must never block each other. Every run also claims the
    dataset key, session-fed or not, so a session-keyed bridge run and a
    dataset-keyed run over the same dataset exclude each other — improves for
    one dataset serialize; an overlapping claim loses and returns ``lock_held``.
    """
    sessions = tuple(
        f"session:{user_id}:{session_id}" for session_id in (session_ids or ()) if session_id
    )
    return (*sessions, f"dataset:{dataset_id}")
