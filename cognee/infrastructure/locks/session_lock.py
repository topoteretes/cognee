"""Per-session lock primitives — in-process asyncio.Lock registry.

Keyed by ``(session_id, op)``. Serializes concurrent asyncio tasks in
the same process that want to mutate the same session.

Scope: single-worker FastAPI deployment (the default). For multi-
worker setups (gunicorn workers, k8s replicas), layer a row-level SQL
advisory lock or Redis SETNX on top — the call sites are already
factored so that's a local change.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Tuple

from cognee.shared.logging_utils import get_logger

logger = get_logger("session_lock")


_locks: dict[Tuple[str, str], asyncio.Lock] = {}
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
async def session_lock(session_id: str, op: str = "write"):
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


async def try_acquire_improve_lock(session_id: str) -> bool:
    """Non-blocking attempt to acquire the improve() lock for a session.

    Returns True if we got it (caller MUST call
    ``release_improve_lock`` when done). Returns False if someone else
    is already improving this session — caller should no-op.
    """
    if not session_id:
        return True  # no-op sessions don't need exclusion

    lock = await _get_lock(session_id, "improve")
    # asyncio.Lock has no try_acquire shortcut pre-3.11; emulate:
    if lock.locked():
        return False
    # acquire() on an unlocked Lock never blocks.
    await lock.acquire()
    return True


async def release_improve_lock(session_id: str) -> None:
    if not session_id:
        return
    lock = await _get_lock(session_id, "improve")
    if lock.locked():
        lock.release()
