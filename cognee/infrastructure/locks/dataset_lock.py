"""Per-dataset lock primitives — in-process asyncio registry.

Serializes operations that mutate the same dataset — pipeline runs
(``add``/``cognify``/``memify``) and delete operations — while letting
different datasets proceed in parallel. Both acquire the lock from the
same registry, so a delete waits for an in-flight pipeline run on the
dataset (and vice versa) and two deletes on the same dataset are
serialized.

NOTE: process-local only (asyncio) — this does NOT protect against multiple
processes/workers operating on the same dataset. To be replaced by a
cross-process mechanism (e.g. DB-backed lock) later.
"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator
from uuid import UUID

_dataset_locks: dict[UUID, asyncio.Lock] = {}
_dataset_locks_guard = asyncio.Lock()

# Tracks the dataset ids whose per-dataset lock is already held by the current
# execution. An operation may legitimately start another operation on the same
# dataset (e.g. cognify_session -> add()/cognify()); without this, re-acquiring
# the non-reentrant _dataset_locks[dataset_id] from the same execution
# self-deadlocks. ContextVar propagates into child tasks spawned via
# asyncio.create_task.
held_datasets: ContextVar[frozenset] = ContextVar("held_datasets", default=frozenset())


async def get_dataset_lock(dataset_id: UUID) -> asyncio.Lock:
    """Return the asyncio.Lock for a dataset, creating it on first use."""
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
