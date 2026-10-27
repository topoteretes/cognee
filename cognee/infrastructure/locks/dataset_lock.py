"""Per-dataset lock primitives — in-process, event-loop-agnostic registry.

Serializes operations that mutate the same dataset — pipeline runs
(``add``/``cognify``/``memify``) and delete operations — while letting
different datasets proceed in parallel. Both acquire the lock from the
same registry, so a delete waits for an in-flight pipeline run on the
dataset (and vice versa) and two deletes on the same dataset are
serialized.

The registry hands out :class:`LoopAgnosticLock` objects, not
``asyncio.Lock``: the cached lock outlives any single event loop, and an
``asyncio.Lock`` binds to the first loop that awaits it (see
``loop_agnostic_lock.py`` for the failure modes). The loop-agnostic lock
also genuinely serializes same-dataset operations across loops/threads,
which the asyncio registry only pretended to do.

LOCK ORDERING (SDK-483): the canonical order is dataset lock FIRST, then the
DatasetQueue slot (``set_database_global_context_variables``). A task that
already holds a queue slot must never wait on a dataset lock — slots are a
global, finite resource (semaphore), so slot-holding lock-waiters can exhaust
them and deadlock the whole process against lock-holding slot-waiters.
``get_dataset_lock`` enforces this: same-dataset violations raise, cross-dataset
ones log a warning.

NOTE: process-local only — this does NOT protect against multiple
processes/workers operating on the same dataset. To be replaced by a
cross-process mechanism (e.g. DB-backed lock) later.
"""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from uuid import UUID

from cognee.shared.logging_utils import get_logger

from .loop_agnostic_lock import LoopAgnosticLock

logger = get_logger("dataset_lock")


class DatasetLock(LoopAgnosticLock):
    """The per-dataset mutual exclusion handed out by :func:`get_dataset_lock`."""


# Plain dict guarded by a threading.Lock (held only for dict ops, never while
# waiting on a dataset lock), so registry access is safe from any loop/thread.
_dataset_locks: dict[UUID, DatasetLock] = {}
_dataset_locks_guard = threading.Lock()


def _check_lock_slot_order(dataset_id: UUID) -> None:
    """Fail fast on the slot->lock order inversion instead of deadlocking.

    A 56h production hang (SDK-483) came from tasks acquiring a DatasetQueue
    slot and then waiting on a dataset lock while lock holders waited for
    slots. Same-dataset inversions have no legitimate caller, so they raise;
    a slot held for a different dataset can still form a cross-dataset cycle,
    so it is loudly logged.
    """
    # Imported lazily: the queue package imports engine-cache modules that must
    # not load at lock-module import time.
    from cognee.infrastructure.databases.dataset_queue import dataset_queue

    held = dataset_queue().current_task_slot_dataset_ids()
    if not held:
        return
    if str(dataset_id) in held:
        raise RuntimeError(
            f"Lock-order inversion: current task holds a DatasetQueue slot for dataset "
            f"{dataset_id} and is trying to acquire that dataset's lock. Canonical order "
            "is dataset lock -> queue slot; close the set_database_global_context_variables "
            "scope before acquiring the lock (see SDK-483)."
        )
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


async def get_dataset_lock(dataset_id: UUID) -> DatasetLock:
    """Return the lock for a dataset, creating it on first use.

    The returned :class:`DatasetLock` is loop-agnostic: it may be created on
    one event loop and acquired on another (e.g. sequential ``asyncio.run``
    calls, or worker threads each running their own loop).
    """
    _check_lock_slot_order(dataset_id)
    with _dataset_locks_guard:
        lock = _dataset_locks.get(dataset_id)
        if lock is None:
            lock = DatasetLock()
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
