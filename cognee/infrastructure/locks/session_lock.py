"""Per-session lock primitives — in-process asyncio registry.

Two primitives:

* ``session_lock(session_id, op)`` — async context manager that
  serializes concurrent tasks on the same ``(session_id, op)`` key.
  Used for short read-modify-write flows (``update_qa``,
  ``add_feedback``, ``delete_qa``).

* ``try_acquire_improve_lock(session_id)`` /
  ``release_improve_lock(session_id)`` — non-blocking claim for
  long-running ``improve()`` calls. The claim is atomic: a
  registry-wide ``asyncio.Lock`` protects a set of held keys, and
  the check-and-add happens inside that critical section so two
  callers can't both see "free" and both think they won.

Scope: single-worker FastAPI. For multi-worker deployments, layer a
row-level SQL advisory lock or Redis SETNX on top — the call sites
are factored so that's a local change.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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


# ----- Non-blocking improve-lock claim ---------------------------------------
#
# asyncio.Lock has no ``acquire_nowait`` on current Python, and the
# obvious ``if lock.locked(): ... await lock.acquire()`` pattern is
# racy — two coroutines can both observe "free" at the check before
# either reaches the acquire. Use a plain set guarded by a
# registry-wide lock instead: the check-and-add happens inside the
# registry lock's critical section, so the test is atomic.

_improving_sessions: set[str] = set()
_improve_registry_lock = asyncio.Lock()


async def try_acquire_improve_lock(session_id: str) -> bool:
    """Atomically claim the improve-lock for ``session_id``.

    Returns ``True`` iff we got it. The caller MUST call
    ``release_improve_lock`` when done (use try/finally). Returns
    ``False`` immediately when another task already holds the lock —
    callers should no-op rather than wait.
    """
    if not session_id:
        return True  # no-op sessions don't need exclusion

    async with _improve_registry_lock:
        if session_id in _improving_sessions:
            return False
        _improving_sessions.add(session_id)
        return True


async def release_improve_lock(session_id: str) -> None:
    """Release the improve-lock for ``session_id``. Idempotent."""
    if not session_id:
        return
    async with _improve_registry_lock:
        _improving_sessions.discard(session_id)
