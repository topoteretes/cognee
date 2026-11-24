"""Per-dataset lock for the stored graph schema — in-process, loop-agnostic registry.

``dataset_schema_lock(dataset_id)`` serializes the writers of one dataset's
configuration row: ``PATCH /datasets/{id}/schema`` is a read-merge-write and
``PUT`` an insert-or-update, and two of them interleaving on one dataset lose a
writer's change silently or collide on the row's unique ``dataset_id`` when
neither finds a row.

This is deliberately not :func:`dataset_lock`, which serializes against running
pipelines: a schema edit must not wait behind an in-flight cognify.

Process-local, like the other locks in this package: workers of one deployment
still race. A version column with a conditional update is the step up if that
is ever needed.
"""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from .loop_agnostic_lock import LoopAgnosticLock

# Guarded by a threading.Lock held only for the dict lookup, never while waiting.
_schema_locks: dict[UUID, LoopAgnosticLock] = {}
_schema_locks_guard = threading.Lock()


def get_dataset_schema_lock(dataset_id: UUID) -> LoopAgnosticLock:
    """Return the schema lock for a dataset, creating it on first use."""
    with _schema_locks_guard:
        return _schema_locks.setdefault(dataset_id, LoopAgnosticLock())


@asynccontextmanager
async def dataset_schema_lock(dataset_id: UUID) -> AsyncIterator[None]:
    """Hold the dataset's schema lock for the duration of the block."""
    async with get_dataset_schema_lock(dataset_id):
        yield
