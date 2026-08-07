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
torn down via :meth:`DatasetQueue._teardown_subprocess_engines`.  This
makes nested ``async with set_database_global_context_variables(D, u)`` scopes
safe — an inner exit never steals an outer holder's slot.

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
        # Pending-close latches — "who must wait, per dataset": ds_key -> future
        # that resolves when a backgrounded engine teardown for that dataset has
        # finished. The next acquirer of the same dataset awaits it before
        # proceeding, so file locks are provably free before a fresh engine
        # opens the same files. Created when a close is scheduled; resolved and
        # removed in the close task's ``finally``, success or failure alike.
        # (Same pending-close idea closing_lru_cache uses internally.)
        self._pending_closes: Dict[str, asyncio.Future] = {}
        # Live close tasks — "which closes are still running, so they can't be
        # lost". asyncio holds only weak references to tasks: a fire-and-forget
        # close with no strong reference can be garbage-collected mid-flight,
        # never finishing the close and never opening its latch. Entries are
        # removed by the task's done-callback (which also surfaces failures);
        # tests and shutdown drain this set to wait for in-flight closes.
        self._background_closes: Set[asyncio.Task] = set()
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

        # If a backgrounded teardown for this dataset is still closing its
        # engines, wait for it: the whole point of the latch is that the
        # response path no longer waits, so the (rare) next acquirer must.
        pending = self._pending_closes.get(ds_key)
        if pending is not None:
            await pending

        # Acquire a fresh slot for this (task, dataset).
        logger.debug("Task %d acquiring dataset queue slot for dataset_id=%s", task_id, dataset_id)
        await self._semaphore.acquire()
        release = _make_release(self._semaphore)

        self._ensure_task_cleanup_registered(task, task_id)
        # After registration, the task entry exists in ``_task_slots``.
        self._task_slots[task_id][ds_key] = SlotEntry(release, depth=1)

    # ----------------------------------------- subprocess engine teardown
    def _schedule_background_teardown(self, ds_key: str) -> None:
        """Detach engines now; close them off the response path.

        Why this exact split, and not something simpler:

        * **Why not await the whole teardown inline** (the original design):
          ``engine.close()`` takes 1.5–2 s per query, and it ran while the
          caller's response waited — measured as roughly half of every serial
          recall's latency. The teardown only fires when *no other task holds
          the dataset*, which is precisely when nobody benefits from waiting.
        * **Why the eviction still happens synchronously here** and only the
          close is backgrounded: while a dying engine is still in the LRU,
          any caller — even the same task, one line after context exit — can
          fetch it, and the background close then kills it mid-use
          ("LadybugAdapter is closed; a new adapter must be created", which
          broke nine e2e suites when the whole teardown was deferred).
          Eviction is cheap and in-memory; after this line every caller
          builds a fresh engine.
        * **Why the pending-close latch**: a fresh engine must not open the
          same database files while the old engine's close is still
          releasing their locks. The next ``ensure_slot`` for this dataset
          awaits the latch — moving the wait from the response (never
          load-bearing there) to the only place it matters.

        The latch always opens; a close failure is surfaced at ERROR with its
        traceback from the task's done-callback (the request this close
        belonged to has already returned, so there is no caller to raise
        into).
        """
        engines = self._detach_subprocess_engines()
        if not engines:
            return

        loop = asyncio.get_running_loop()
        latch: asyncio.Future = loop.create_future()
        self._pending_closes[ds_key] = latch

        async def _close() -> None:
            # No exception is expected here, so none is caught: the latch must
            # open either way (a failed close must not brick the dataset), and
            # the failure itself is surfaced below with its full traceback.
            try:
                await self._close_engines(engines)
            finally:
                latch.set_result(None)
                if self._pending_closes.get(ds_key) is latch:
                    del self._pending_closes[ds_key]

        task = loop.create_task(_close())
        self._background_closes.add(task)

        def _surface(done: asyncio.Task) -> None:
            self._background_closes.discard(done)
            if not done.cancelled() and done.exception() is not None:
                # The request this teardown belonged to has already returned,
                # so there is no caller to raise into — ERROR with the stack is
                # the loudest honest channel. If the close genuinely left file
                # locks behind, the next engine open fails loudly in a real
                # request; nothing is silently lost.
                exc = done.exception()
                logger.error(
                    "Background engine teardown failed for %s",
                    ds_key,
                    # The structlog exception processor only unpacks a real
                    # (type, value, traceback) tuple; anything else falls back
                    # to sys.exc_info(), which is empty in a done-callback.
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(_surface)

    def _detach_subprocess_engines(self) -> list:
        """Synchronously evict subprocess-mode engines from their caches.

        This MUST stay synchronous and run before the release returns: eviction
        is what guarantees no later caller can fetch a dying engine from the
        LRU (they build a fresh one instead). Only the expensive
        ``engine.close()`` of the returned engines may be deferred.

        Reads the current task's ContextVar-based graph/vector config to
        identify which cached engines to detach.  Lazy imports avoid circular
        dependencies at module load time.
        """
        from cognee.infrastructure.databases.graph.config import get_graph_context_config
        from cognee.infrastructure.databases.vector.config import get_vectordb_context_config

        detached = []

        g_cfg = get_graph_context_config()
        if g_cfg.get("graph_database_subprocess_enabled"):
            from cognee.infrastructure.databases.graph.get_graph_engine import (
                create_graph_engine,
                evict_graph_engine,
                is_graph_engine_cached,
            )

            if is_graph_engine_cached(**g_cfg):
                detached.append(create_graph_engine(**g_cfg))
                evict_graph_engine(**g_cfg)

        v_cfg = get_vectordb_context_config()
        if v_cfg.get("vector_db_subprocess_enabled"):
            from cognee.infrastructure.databases.vector.create_vector_engine import (
                create_vector_engine,
                evict_vector_engine,
                is_vector_engine_cached,
            )

            if is_vector_engine_cached(**v_cfg):
                detached.append(create_vector_engine(**v_cfg))
                evict_vector_engine(**v_cfg)

        return detached

    async def _close_engines(self, engines: list) -> None:
        """Close detached engines so their DB file locks are released."""
        for engine in engines:
            if hasattr(engine, "close"):
                await engine.close()

    async def _teardown_subprocess_engines(self) -> None:
        """Detach and close subprocess-mode engines in one awaited step.

        Used where there is no task to background the close onto (and by
        callers/tests that want the full synchronous-contract teardown).
        """
        await self._close_engines(self._detach_subprocess_engines())

    # -------------------------------------------------------- release_slot_for
    async def release_slot_for(self, dataset_id: Any = None) -> None:
        """Decrement this task's depth counter for ``dataset_id`` and release
        the semaphore slot when the counter reaches zero.

        When this is the very last holder across all tasks for
        ``dataset_id``, subprocess engines are torn down via
        :meth:`_teardown_subprocess_engines` while the semaphore slot is
        still held so that no new operation can observe a half-torn-down
        resource.  The slot is freed afterwards regardless of whether the
        teardown succeeds or raises.

        No-op when the queue is disabled, there is no running task, or the
        current task does not hold a slot for ``dataset_id``.
        """
        if not self._enabled:
            return

        task = asyncio.current_task()
        if task is None:
            await self._teardown_subprocess_engines()
            return

        task_id = id(task)
        ds_key = f"ds:{dataset_id}" if dataset_id is not None else "ds:<none>"

        entry = self._task_slots.get(task_id, {}).get(ds_key)
        if entry is None:
            return

        entry.depth -= 1
        if entry.depth > 0:
            return

        # About to fully release.  Detach (evict) subprocess engines only when
        # no other task holds the same dataset. The eviction itself is
        # synchronous — after this line no caller can fetch the dying engines
        # from the cache; they build fresh ones. Only the expensive
        # ``engine.close()`` runs in the background: the caller's response
        # must not wait on the close of engines it has already finished
        # using. The pending-close latch makes the next ``ensure_slot`` for
        # the same dataset wait for that close before spawning fresh engines
        # against the same database files.
        try:
            other_holds = any(
                ds_key in slots for tid, slots in self._task_slots.items() if tid != task_id
            )
            if not other_holds:
                self._schedule_background_teardown(ds_key)
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
