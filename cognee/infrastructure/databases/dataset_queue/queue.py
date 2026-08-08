"""Semaphore-backed dataset queue with per-task, per-dataset slot tracking.

Each distinct dataset a task touches via :func:`set_database_global_context_variables`
(which calls :meth:`DatasetQueue.ensure_slot` under the hood) takes its own
slot against the shared budget.

Ref-counting model (per (task, dataset)):

Repeated :meth:`DatasetQueue.ensure_slot` calls for the same ``(task, dataset)``
bump a per-entry depth counter rather than re-acquiring the semaphore. The
corresponding :meth:`DatasetQueue.release_slot_for` (async) decrements that
counter; the underlying semaphore slot is freed only when the counter hits
zero.  When the last holder across all tasks exits, subprocess engines are
evicted from the engine caches via
:meth:`DatasetQueue._evict_subprocess_engines`; the caches close them off the
response path.  This makes nested
``async with set_database_global_context_variables(D, u)`` scopes safe — an
inner exit never steals an outer holder's slot.

Task-end cleanup is a safety net: when the current task finishes, every entry
still in ``_task_slots`` is force-released regardless of depth. This covers
``await``-style callers that never decrement, so long-lived task slots still
clean up correctly.

Re-entrancy rules:

* Same task + same dataset → depth counter increments; no new acquire.
* Same task + different dataset → acquire an additional slot (may block).
* Different task (e.g. a child task that inherited the ContextVar) → treated
  as a fresh task; acquires its own independent slot.

Configuration:

* ``DATASET_QUEUE_ENABLED`` — env var. Truthy values enable the queue.
* ``DATASET_QUEUE_MAX_CONCURRENT`` — env var. Defaults to ``DATABASE_MAX_LRU_CACHE_SIZE`` for a safe baseline
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Set

from cognee.shared.lru_cache import DATABASE_MAX_LRU_CACHE_SIZE
from cognee.shared.logging_utils import get_logger

logger = get_logger("DatasetQueue")


# Recognised truthy values for ``DATASET_QUEUE_ENABLED``.
TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})


class DatasetQueueSettings:
    """Effective runtime settings for the dataset queue."""

    __slots__ = ("enabled", "max_concurrent")

    def __init__(self, enabled: bool, max_concurrent: int) -> None:
        self.enabled = enabled
        self.max_concurrent = max_concurrent


def get_dataset_queue_settings() -> DatasetQueueSettings:
    """Return effective settings. Test mock seam."""
    raw = os.getenv("DATASET_QUEUE_ENABLED", "true").strip().lower()
    enabled = raw in TRUE_VALUES

    max_concurrent = os.getenv("DATASET_QUEUE_MAX_CONCURRENT", None)
    if not max_concurrent:
        # Default to the same max concurrency as the LRU cache size, which is a reasonable baseline for a shared resource limit.
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


class SlotEntry:
    """A single acquired slot with a nesting depth counter."""

    __slots__ = ("release", "depth")

    def __init__(self, release: Callable[[], None], depth: int = 1) -> None:
        self.release = release
        self.depth = depth


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

        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(safe_max)

        # Per-task slot registry: task_id → { slot_key → SlotEntry }.
        # ``slot_key`` is ``"ds:<dataset_id>"`` for ``ensure_slot`` and
        # ``"acquire:<unique>"`` for ``acquire()``. A task may hold multiple
        # entries; all are released together when the task finishes.
        self._task_slots: Dict[int, Dict[str, SlotEntry]] = {}
        # Track which tasks already have a done-callback registered so we
        # don't register multiple cleanup handlers for a single task.
        self._registered_tasks: Set[int] = set()

    # ------------------------------------------------------ active datasets
    def active_dataset_ids(self) -> set:
        """Dataset ids (as strings) currently holding at least one slot.

        Read-only snapshot for engine-cache pinning: capacity eviction must
        not close an engine whose dataset an admitted pipeline is still
        using. Safe to call from any thread — iterates over ``list()``
        snapshots, so concurrent slot changes on the event loop can't break
        the iteration; the result is advisory-fresh by nature.
        """
        if not self._enabled:
            return set()
        active = set()
        for slots in list(self._task_slots.values()):
            for slot_key in list(slots):
                # "ds:<none>" tracks dataset-less operations; it can never
                # correspond to a database name, so don't report it.
                if slot_key.startswith("ds:") and slot_key != "ds:<none>":
                    active.add(slot_key[3:])
        return active

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
            for entry in slots.values():
                # Backstop: release whatever's left regardless of depth.
                entry.release()

        task.add_done_callback(_release_all_on_done)

    # ------------------------------------------------------------ ensure_slot
    async def ensure_slot(self, dataset_id) -> None:
        """Acquire (or bump the depth of) a slot for (current task, ``dataset_id``).

        Rules:

        * If the current task already has an entry for the same dataset →
          increment its depth counter; do **not** re-acquire the semaphore.
        * Otherwise → acquire a fresh slot; it will be released when the
          matching :meth:`release_slot_for` drops the counter to zero, or
          unconditionally at task-end as a backstop.

        This is the mechanism behind
        :func:`cognee.context_global_variables.set_database_global_context_variables`:
        every call there passes through here.
        """
        if not self._enabled:
            return

        task = asyncio.current_task()
        if task is None:
            # Rare: no running task, no way to track ownership.
            raise RuntimeError("DatasetQueue.ensure_slot called outside of a running task")

        task_id = id(task)
        ds_key = f"ds:{dataset_id}" if dataset_id is not None else "ds:<none>"

        entry = self._task_slots.get(task_id, {}).get(ds_key)
        if entry is not None:
            # Same task, same dataset — re-entrant: bump depth, do NOT re-acquire.
            entry.depth += 1
            return

        # Acquire a fresh slot for this (task, dataset).
        logger.debug("Task %d acquiring dataset queue slot for dataset_id=%s", task_id, dataset_id)
        await self._semaphore.acquire()
        release = _make_release(self._semaphore)

        self._ensure_task_cleanup_registered(task, task_id)
        # After registration, the task entry exists in ``_task_slots``.
        self._task_slots[task_id][ds_key] = SlotEntry(release, depth=1)

    # ----------------------------------------- subprocess engine teardown
    def _evict_subprocess_engines(self) -> None:
        """Evict this context's subprocess-mode engines from their caches.

        Eviction is the queue's entire teardown responsibility — the engine
        cache owns the close lifecycle, and it already gives the release path
        everything it needs:

        * a subprocess-backed adapter's close runs on the cache's dedicated
          close threads, so it makes progress even while this loop's thread is
          blocked. (An earlier revision scheduled the close as a task on this
          loop; a synchronous engine spawn right after release then starved it
          and exhausted the worker's bounded open-retry — the "Lock is held by
          PID N" e2e failures.)
        * the close is registered in the cache's pending-close registry, so
          the next creation of the same engine — through the queue or a direct
          ``get_graph_engine()`` — waits for the old worker to exit before
          opening the same database files, with the worker's open-retry as the
          backstop for the residual window.
        * a close on those threads survives event-loop teardown, so
          sequential ``asyncio.run()`` phases in one process are safe.

        The eviction is ``force_close=True``: the adapter closes now even if
        an idle holder still pins its lease proxy (e.g. a test keeping a
        ``get_graph_engine()`` handle across pipeline calls). A deferred
        lease-close would keep the worker — and its file locks — alive
        indefinitely, and the next engine open for the dataset would exhaust
        its retries against a lock that never frees. Holders re-resolve on
        next use; nothing is mid-query here because teardown only fires when
        no other task holds the dataset.

        Closing engines here instead re-implements that machinery one level
        up, loop-bound and invisible to the registry: fetching a lease proxy
        in order to close it defers the real close behind the lease and hides
        it from creators.

        Eviction runs synchronously on the release path on purpose: a dying
        engine left in the LRU can be fetched by the very next caller and then
        die mid-use ("LadybugAdapter is closed; a new adapter must be
        created", which broke nine e2e suites when the whole teardown was
        deferred). After eviction every caller builds a fresh engine.

        Reads the current task's ContextVar-based graph/vector config to
        identify which cached engines to evict. Lazy imports avoid circular
        dependencies at module load time.
        """
        from cognee.infrastructure.databases.graph.config import get_graph_context_config
        from cognee.infrastructure.databases.vector.config import get_vectordb_context_config

        g_cfg = get_graph_context_config()
        if g_cfg.get("graph_database_subprocess_enabled"):
            from cognee.infrastructure.databases.graph.get_graph_engine import evict_graph_engine

            evict_graph_engine(force_close=True, **g_cfg)

        v_cfg = get_vectordb_context_config()
        if v_cfg.get("vector_db_subprocess_enabled"):
            from cognee.infrastructure.databases.vector.create_vector_engine import (
                evict_vector_engine,
            )

            evict_vector_engine(force_close=True, **v_cfg)

    # -------------------------------------------------------- release_slot_for
    async def release_slot_for(self, dataset_id: Any = None) -> None:
        """Decrement this task's depth counter for ``dataset_id`` and release
        the semaphore slot when the counter reaches zero.

        When this is the very last holder across all tasks for
        ``dataset_id``, subprocess engines are evicted via
        :meth:`_evict_subprocess_engines` while the semaphore slot is still
        held, so no new operation can fetch a dying engine from the cache.
        The slot is freed afterwards regardless of whether the eviction
        succeeds or raises.

        No-op when the queue is disabled, there is no running task, or the
        current task does not hold a slot for ``dataset_id``.
        """
        if not self._enabled:
            return

        task = asyncio.current_task()
        if task is None:
            self._evict_subprocess_engines()
            return

        task_id = id(task)
        ds_key = f"ds:{dataset_id}" if dataset_id is not None else "ds:<none>"

        entry = self._task_slots.get(task_id, {}).get(ds_key)
        if entry is None:
            return

        entry.depth -= 1
        if entry.depth > 0:
            return

        # About to fully release.  Evict subprocess engines only when no other
        # task holds the same dataset; the engine cache closes them off the
        # response path (see ``_evict_subprocess_engines`` for the full
        # contract). The caller's response must not wait on the close of
        # engines it has already finished using.
        try:
            other_holds = any(
                ds_key in slots for tid, slots in self._task_slots.items() if tid != task_id
            )
            if not other_holds:
                self._evict_subprocess_engines()
        finally:
            self._task_slots.get(task_id, {}).pop(ds_key, None)
            logger.debug(
                "Task %d releasing dataset queue slot for dataset_id=%s",
                task_id,
                dataset_id,
            )
            entry.release()

    # ---------------------------------------------------------------- acquire
    @asynccontextmanager
    async def acquire(self):
        """Scoped slot for call sites without a natural dataset id.

        Used by ``visualize_graph`` and the access-control-disabled search
        branch — neither has a per-call dataset to key on.

        Re-entrant: if the current task already holds *any* slot (via
        ``ensure_slot`` or a prior ``acquire``), this is a pass-through.
        Otherwise a fresh slot is taken and released at block exit.

        Unlike ``ensure_slot``/``release_slot_for``, ``acquire`` is always
        strictly scoped (enter/exit pair in a single ``async with``), so no
        depth counter is needed — the entry is popped and released on exit.
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
            # Store a SlotEntry for type uniformity; acquire() is scoped
            # and doesn't use the depth counter.
            self._task_slots[task_id][slot_key] = SlotEntry(release, depth=1)

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
