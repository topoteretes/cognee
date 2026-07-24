"""Cross-worker locking for pipeline runs that target the same dataset."""

import asyncio
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy import text

from cognee.infrastructure.databases.relational import get_relational_engine


_SQLITE_LOCK_STRIPES = 256
_SQLITE_LOCK_POLL_INTERVAL_SECONDS = 0.05


@dataclass
class _LocalLockEntry:
    lock: asyncio.Lock
    users: int = 0


# Protecting this small registry with a threading lock avoids binding the guard to
# one event loop. The registry key includes the loop because asyncio locks cannot
# safely be shared between loops (which is common in SDK users and unit tests).
_local_locks: dict[tuple[asyncio.AbstractEventLoop, UUID], _LocalLockEntry] = {}
_local_locks_guard = threading.Lock()


def _postgres_lock_key(dataset_id: UUID) -> int:
    """Return a deterministic signed bigint key namespaced to pipeline locks."""
    digest = sha256(b"cognee:dataset-pipeline:" + dataset_id.bytes).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


def _sqlite_lock_path(engine, dataset_id: UUID) -> Path | None:
    """Return one of a bounded number of lock files next to the SQLite database."""
    database = engine.url.database
    if not database or database == ":memory:":
        return None

    # A fixed number of stripes prevents an unbounded lock-file leak. A hash
    # collision only serializes two unrelated datasets; it cannot compromise
    # same-dataset exclusion.
    stripe = sha256(dataset_id.bytes).digest()[0] % _SQLITE_LOCK_STRIPES
    database_path = Path(os.path.abspath(database))
    return database_path.parent / f".cognee-pipeline-{stripe:02x}.lock"


@asynccontextmanager
async def _local_dataset_lock(dataset_id: UUID) -> AsyncGenerator[None, None]:
    """Serialize same-process callers and remove unused registry entries."""
    loop = asyncio.get_running_loop()
    key = (loop, dataset_id)
    with _local_locks_guard:
        entry = _local_locks.get(key)
        if entry is None:
            entry = _LocalLockEntry(asyncio.Lock())
            _local_locks[key] = entry
        entry.users += 1

    try:
        async with entry.lock:
            yield
    finally:
        with _local_locks_guard:
            entry.users -= 1
            if entry.users == 0:
                _local_locks.pop(key, None)


@asynccontextmanager
async def _sqlite_advisory_lock(engine, dataset_id: UUID) -> AsyncGenerator[None, None]:
    """Use a host-wide OS advisory lock for a file-backed SQLite database."""
    lock_path = _sqlite_lock_path(engine, dataset_id)
    if lock_path is None:
        # In-memory SQLite databases cannot be shared between worker processes.
        yield
        return

    from filelock import FileLock, Timeout

    lock = FileLock(lock_path, thread_local=False)
    while True:
        try:
            # timeout=0 performs one non-blocking OS lock attempt. Polling with
            # asyncio avoids retaining one worker thread for every long-running
            # pipeline and makes cancellation immediate and leak-free.
            lock.acquire(timeout=0)
            break
        except Timeout:
            await asyncio.sleep(_SQLITE_LOCK_POLL_INTERVAL_SECONDS)

    try:
        yield
    finally:
        lock.release()


@asynccontextmanager
async def _cross_worker_dataset_lock(dataset_id: UUID) -> AsyncGenerator[None, None]:
    """Acquire the database-appropriate cross-worker lock for ``dataset_id``."""
    engine = get_relational_engine().engine
    if engine.dialect.name == "postgresql":
        lock_key = _postgres_lock_key(dataset_id)
        async with engine.connect() as connection:
            await connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": lock_key})
            # Do not hold an idle transaction for the lifetime of a pipeline run.
            await connection.commit()
            try:
                yield
            finally:
                await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key})
                await connection.commit()
        return

    if engine.dialect.name == "sqlite":
        async with _sqlite_advisory_lock(engine, dataset_id):
            yield
        return

    # Unknown relational backends retain process-local safety. SQLAlchemy
    # adapters currently support PostgreSQL and SQLite, so this is defensive.
    yield


@asynccontextmanager
async def dataset_pipeline_lock(dataset_id: UUID) -> AsyncGenerator[None, None]:
    """Serialize pipeline runs for a dataset across tasks and API workers.

    The cheap in-process lock is acquired first, so only one task per worker
    contends for the cross-worker resource. PostgreSQL advisory locks coordinate
    across hosts; file-backed SQLite uses host-wide file locks.
    """
    async with _local_dataset_lock(dataset_id):
        async with _cross_worker_dataset_lock(dataset_id):
            yield
