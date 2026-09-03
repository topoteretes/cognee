"""Per-dataset lock primitives — in-process asyncio registry.

Serializes operations that mutate the same dataset — pipeline runs
(``add``/``cognify``/``memify``) and delete operations — while letting
different datasets proceed in parallel. Both acquire the lock from the
same registry, so a delete waits for an in-flight pipeline run on the
dataset (and vice versa) and two deletes on the same dataset are
serialized.

LOCK ORDERING (SDK-483): the canonical order is dataset lock FIRST, then the
DatasetQueue slot (``set_database_global_context_variables``). A task that
already holds a queue slot must never wait on a dataset lock — slots are a
global, finite resource (semaphore), so slot-holding lock-waiters can exhaust
them and deadlock the whole process against lock-holding slot-waiters.
``get_dataset_lock`` logs violations at acquisition time (it does not raise:
the deprecated ``await set_database_global_context_variables`` form holds a
task-lifetime slot by design, so pre-existing callers of it must keep working).

NOTE: process-local only (asyncio) — this does NOT protect against multiple
processes/workers operating on the same dataset. To be replaced by a
cross-process mechanism (e.g. DB-backed lock) later.
"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator
from uuid import UUID

from cognee.shared.logging_utils import get_logger

logger = get_logger("dataset_lock")

_dataset_locks: dict[UUID, asyncio.Lock] = {}
_dataset_locks_guard = asyncio.Lock()


def _check_lock_slot_order(dataset_id: UUID) -> None:
    """Log the slot->lock order inversion that deadlocked SDK-483.

    A 56h production hang (SDK-483) came from tasks acquiring a DatasetQueue
    slot and then waiting on a dataset lock while lock holders waited for
    slots. Detection only — no raise: the deprecated ``await
    set_database_global_context_variables`` form legitimately holds a
    task-lifetime slot (it also pins the dataset's engines open), so existing
    callers of it must keep working. The log is deterministic on every
    slot->lock acquisition, so a reintroduced inversion is visible in dev and
    CI logs long before it deadlocks under load.
    """
    # Imported lazily: the queue package imports engine-cache modules that must
    # not load at lock-module import time.
    from cognee.infrastructure.databases.dataset_queue import dataset_queue

    held = dataset_queue().current_task_slot_dataset_ids()
    if not held:
        return
    if str(dataset_id) in held:
        logger.error(
            "Lock-order inversion: current task holds a DatasetQueue slot for dataset %s "
            "and is acquiring that dataset's lock. Canonical order is dataset lock -> "
            "queue slot; under queue pressure this shape deadlocks the process (SDK-483). "
            "Close the set_database_global_context_variables scope before taking the lock.",
            dataset_id,
        )
        return
    logger.warning(
        "Task holds DatasetQueue slot(s) for dataset(s) %s while acquiring the lock for "
        "dataset %s. Canonical order is dataset lock -> queue slot; this risks a "
        "cross-dataset deadlock (see SDK-483).",
        sorted(held),
        dataset_id,
    )


# Tracks the dataset ids whose per-dataset lock is already held by the current
# execution. An operation may legitimately start another operation on the same
# dataset (e.g. cognify_session -> add()/cognify()); without this, re-acquiring
# the non-reentrant _dataset_locks[dataset_id] from the same execution
# self-deadlocks. ContextVar propagates into child tasks spawned via
# asyncio.create_task.
held_datasets: ContextVar[frozenset] = ContextVar("held_datasets", default=frozenset())


async def get_dataset_lock(dataset_id: UUID) -> asyncio.Lock:
    """Return the asyncio.Lock for a dataset, creating it on first use."""
    _check_lock_slot_order(dataset_id)
    async with _dataset_locks_guard:
        lock = _dataset_locks.get(dataset_id)
        if lock is None:
            lock = asyncio.Lock()
            _dataset_locks[dataset_id] = lock
        return lock


@asynccontextmanager
async def dataset_lock(dataset_id: UUID) -> AsyncIterator[None]:
    """Hold the per-dataset lock for the duration of the block.

    Re-entrant per execution context: when the current execution already holds
    the dataset's lock (see ``held_datasets``), the block runs without
    re-acquiring — external operations stay excluded by the lock the ancestor
    holds.

    Usage::

        async with dataset_lock(dataset_id):
            ...  # mutate the dataset
    """
    if dataset_id in held_datasets.get():
        yield
        return

    async with await get_dataset_lock(dataset_id):
        token = held_datasets.set(held_datasets.get() | {dataset_id})
        try:
            yield
        finally:
            held_datasets.reset(token)
