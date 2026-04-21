"""Semaphore-backed dataset queue with per-task, per-dataset slot tracking.

Each distinct dataset a task touches via :func:`set_database_global_context_variables`
(which calls :meth:`DatasetQueue.ensure_slot` under the hood) takes its own
slot against the shared budget. Slots accumulate across the lifetime of the
task and are released *together* when the task completes — normal return,
exception, or cancellation.

Re-entrancy rules:

* Same task + same dataset → no-op. The slot is already held.
* Same task + different dataset → acquire an additional slot (may block).
* Different task (e.g. a child task that inherited the ContextVar) → treated
  as a fresh task; acquires its own independent slot.

Configuration:

* ``DATASET_QUEUE_ENABLED`` — env var. Truthy values enable the queue.
* ``DATABASE_MAX_LRU_CACHE_SIZE`` — shared constant from
  :mod:`cognee.shared.lru_cache` (default: ``128``).

Sizing note:
    A single task touching K distinct datasets holds K slots concurrently.
    For the default ``max_concurrent=128`` this accommodates typical calls
    (1–3 datasets per operation) with large headroom. If you design a flow
    that fans out many datasets within a single task, size the budget
    accordingly.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Set

from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE


# Recognised truthy values for ``DATASET_QUEUE_ENABLED``.
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})


class DatasetQueueSettings:
    """Effective runtime settings for the dataset queue."""

    __slots__ = ("enabled", "max_concurrent")

    def __init__(self, enabled: bool, max_concurrent: int) -> None:
        # TODO: Return commented out settings
        # self.enabled = enabled
        # self.max_concurrent = max_concurrent
        self.enabled = True
        self.max_concurrent = (
            3  # TESTING CI PURPOSES ONLY, REMOVE THIS AND UNCOMMENT ABOVE BEFORE MERGING
        )


def get_dataset_queue_settings() -> DatasetQueueSettings:
    """Return effective settings. Test mock seam."""
    raw = os.getenv("DATASET_QUEUE_ENABLED", "").strip().lower()
    enabled = raw in _TRUTHY_VALUES

    max_concurrent = os.getenv("DATASET_QUEUE_MAX_CONCURRENT", None)
    if not max_concurrent:
        max_concurrent = int(DATABASE_MAX_LRU_CACHE_SIZE)

    return DatasetQueueSettings(enabled=enabled, max_concurrent=max_concurrent)


def _make_release(semaphore: asyncio.Semaphore) -> Callable[[], None]:
    """Build an idempotent releaser for one semaphore acquisition.

    The task-end done-callback and any direct cleanup path both call the
    same returned function; the ``released`` flag guarantees the underlying
    ``semaphore.release()`` fires exactly once.
    """
    released = False

    def _release() -> None:
        nonlocal released
        if not released:
            released = True
            semaphore.release()

    return _release


class DatasetQueue:
    """Concurrency limiter for dataset-level operations.

    When ``enabled`` is ``False`` all methods are pass-throughs.
    """

    def __init__(self, enabled: bool, max_concurrent: int) -> None:
        safe_max = int(max_concurrent)
        if safe_max < 1:
            self._enabled: bool = False
            return

        self._enabled: bool = bool(enabled)
        self._max_concurrent: int = safe_max
        # On Python 3.10+ the semaphore is loop-agnostic until first acquire.
        # On 3.9 it binds eagerly via ``get_event_loop()``; callers should
        # first touch the singleton from inside a running loop.
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(safe_max)

        # Per-task slot registry: task_id → { slot_key → release_fn }.
        # ``slot_key`` is ``"ds:<dataset_id>"`` for ``ensure_slot`` and
        # ``"acquire:<unique>"`` for ``acquire()``. A task may hold multiple
        # entries; all are released together when the task finishes.
        self._task_slots: Dict[int, Dict[str, Callable[[], None]]] = {}
        # Track which tasks already have a done-callback registered so we
        # don't register multiple cleanup handlers for a single task.
        self._registered_tasks: Set[int] = set()

    # ---------------------------------------------------- task cleanup setup
    def _ensure_task_cleanup_registered(self, task: asyncio.Task, task_id: int) -> None:
        """Idempotently register a done-callback that releases all of
        this task's slots when it finishes.
        """
        if task_id in self._registered_tasks:
            # Already registered; ensure the slot dict exists (may have been
            # cleaned up by a previous scope-end release that emptied it).
            self._task_slots.setdefault(task_id, {})
            return
        self._registered_tasks.add(task_id)
        # Create the slot dict for this task up front so callers can write
        # to it immediately after this method returns.
        self._task_slots.setdefault(task_id, {})

        def _release_all_on_done(_t: asyncio.Task, _tid: int = task_id) -> None:
            slots = self._task_slots.pop(_tid, {})
            self._registered_tasks.discard(_tid)
            for release in slots.values():
                release()

        task.add_done_callback(_release_all_on_done)

    # ------------------------------------------------------------ ensure_slot
    async def ensure_slot(self, dataset_id: Any = None) -> None:
        """Acquire a slot for (current task, ``dataset_id``) if not already held.

        Rules:

        * If the current task already has a slot for the same dataset → no-op.
        * Otherwise → acquire a fresh slot; it will be released automatically
          when the task completes, along with any other slots the task
          accumulated.

        This is the mechanism behind
        :func:`cognee.context_global_variables.set_database_global_context_variables`:
        every call there passes through here.
        """
        if not self._enabled:
            return

        task = asyncio.current_task()
        if task is None:
            # Rare: no running task, no way to track ownership.
            # Acquire without automatic release — caller must not rely on it.
            await self._semaphore.acquire()
            return

        task_id = id(task)
        ds_key = f"ds:{dataset_id}" if dataset_id is not None else "ds:<none>"

        entry = self._task_slots.get(task_id)
        if entry is not None and ds_key in entry:
            # Same task, same dataset — re-entrant no-op.
            return

        # Acquire a fresh slot for this (task, dataset).
        await self._semaphore.acquire()
        release = _make_release(self._semaphore)

        self._ensure_task_cleanup_registered(task, task_id)
        # After registration, the task entry exists in ``_task_slots``.
        self._task_slots[task_id][ds_key] = release

    # ---------------------------------------------------------------- acquire
    @asynccontextmanager
    async def acquire(self):
        """Scoped slot for call sites without a natural dataset id.

        Used by ``visualize_graph`` and the access-control-disabled search
        branch — neither has a per-call dataset to key on.

        Re-entrant: if the current task already holds *any* slot (via
        ``ensure_slot`` or a prior ``acquire``), this is a pass-through.
        Otherwise a fresh slot is taken and released at block exit.
        """
        if not self._enabled:
            yield
            return

        task = asyncio.current_task()
        task_id = id(task) if task is not None else None

        # Re-entrant: if this task is already holding at least one slot,
        # don't take another.
        if (
            task_id is not None and self._task_slots.get(task_id)  # non-empty dict
        ):
            yield
            return

        await self._semaphore.acquire()
        release = _make_release(self._semaphore)

        slot_key = None
        if task_id is not None:
            self._ensure_task_cleanup_registered(task, task_id)
            slot_key = f"acquire:{id(release)}"
            self._task_slots[task_id][slot_key] = release

        try:
            yield
        finally:
            # Release on scope exit — we don't wait for task end for these.
            if task_id is not None and slot_key is not None:
                slots = self._task_slots.get(task_id)
                if slots is not None:
                    slots.pop(slot_key, None)
            release()


def dataset_queue() -> DatasetQueue:
    """Return the process-wide :class:`DatasetQueue` singleton."""
    if dataset_queue._instance is not None:  # type: ignore[attr-defined]
        return dataset_queue._instance  # type: ignore[attr-defined]

    settings = get_dataset_queue_settings()
    instance = DatasetQueue(
        enabled=settings.enabled,
        max_concurrent=settings.max_concurrent,
    )
    dataset_queue._instance = instance  # type: ignore[attr-defined]
    return instance


# Singleton storage — reset between tests by the reset_queue_singleton fixture.
dataset_queue._instance = None  # type: ignore[attr-defined]
