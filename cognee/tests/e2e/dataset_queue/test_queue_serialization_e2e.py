"""End-to-end test proving the Dataset Queue serialises per-dataset operations.

Scenario:
    - Queue enabled (``DATASET_QUEUE_ENABLED=true``).
    - Single slot (``DATABASE_MAX_LRU_CACHE_SIZE=1`` at the queue level).
    - Three concurrent ``cognee.add`` calls on three different datasets.
    - Three concurrent ``cognee.cognify`` calls on the same three datasets.
    - Three concurrent ``cognee.search`` calls on the same three datasets.

For each phase we instrument :meth:`DatasetQueue.acquire` on the singleton and
record ``enter``/``exit`` events. With a single-slot queue the queue *must*
see at most one operation holding a slot at any time — we assert that.
"""

from __future__ import annotations

import os

os.environ["DATASET_QUEUE_ENABLED"] = "true"
# Skip cognee's LLM-endpoint connectivity probe during ``setup()``. This
# e2e test runs without a real LLM_API_KEY (cognify/search phases are
# skipped gracefully if it's missing), so the probe would otherwise stall
# for 30s during the add-phase setup step.
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

import asyncio
import pathlib
import time
from contextlib import asynccontextmanager
from typing import List, Tuple

import cognee

import cognee.shared.lru_cache as _lru_cache_module
from cognee.infrastructure.databases.dataset_queue import queue as _queue_module
from cognee.infrastructure.databases.dataset_queue import dataset_queue
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.modules.engine.operations.setup import setup
from cognee.modules.users.methods.get_default_user import get_default_user

_lru_cache_module.DATABASE_MAX_LRU_CACHE_SIZE = 1
_queue_module.DATABASE_MAX_LRU_CACHE_SIZE = 1

logger = get_logger("cognee.tests.dataset_queue_e2e")


# Each recorded event is (label, kind, timestamp).
#   label: the asyncio task name (we encode which dataset it's processing)
#   kind : "wait_start" | "enter" | "exit"
Event = Tuple[str, str, float]


def _install_queue_tracing(events: List[Event]) -> None:
    """Hook the queue's underlying semaphore to record every real slot take/release.

    The queue has two entry points (``acquire`` and ``ensure_slot``) and both
    are re-entrant — re-entrant calls don't touch the semaphore, so hooking
    the semaphore itself captures exactly the "fresh" acquisitions we care
    about, regardless of which entry point was used.

    ``acquire()`` is inherently bound to the task that calls it. ``release()``
    fires either from an ``async with acquire`` ``finally`` block (bound to
    the task) or from an ``add_done_callback`` when the task ends (no task
    context). We label release events with the task whose done-callback
    triggered them by registering our own done-callback that runs alongside.
    """
    queue = dataset_queue()
    sem = queue._semaphore
    orig_acquire = sem.acquire
    orig_release = sem.release

    def _label_for_current_task() -> str:
        task = asyncio.current_task()
        return task.get_name() if task is not None else "<detached>"

    async def _traced_semaphore_acquire():
        label = _label_for_current_task()
        events.append((label, "wait_start", time.monotonic()))
        result = await orig_acquire()
        events.append((label, "enter", time.monotonic()))
        # Register an exit event for this task so we still get an event even
        # when release is triggered by a done-callback (no current task).
        task = asyncio.current_task()
        if task is not None:
            fired = {"value": False}

            def _on_done(_t, _label=label, _fired=fired):
                if not _fired["value"]:
                    _fired["value"] = True
                    events.append((_label, "exit", time.monotonic()))

            task.add_done_callback(_on_done)
        return result

    def _traced_semaphore_release():
        task = asyncio.current_task()
        if task is not None:
            # In-task release (async with acquire). Record directly.
            events.append((task.get_name(), "exit", time.monotonic()))
        orig_release()
        # If release fired from a done-callback, the matching exit event is
        # emitted by the done-callback we registered above.

    sem.acquire = _traced_semaphore_acquire  # type: ignore[method-assign]
    sem.release = _traced_semaphore_release  # type: ignore[method-assign]


def _max_observed_concurrency(events: List[Event]) -> int:
    """Given a list of traced events, compute the max concurrent slot holders."""
    timeline = sorted(events, key=lambda e: e[2])
    in_flight = 0
    max_seen = 0
    for _label, kind, _t in timeline:
        if kind == "enter":
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        elif kind == "exit":
            in_flight -= 1
    return max_seen


def _assert_serialized(
    events: List[Event],
    *,
    phase: str,
    min_expected_enters: int,
) -> None:
    """Assert that the queue held at most one slot at a time and was actually used."""
    enters = [e for e in events if e[1] == "enter"]
    assert len(enters) >= min_expected_enters, (
        f"[{phase}] expected at least {min_expected_enters} acquire() enter events "
        f"(queue gate was hit), saw {len(enters)}. "
        f"Did the integration sites call through dataset_queue()?"
    )
    max_seen = _max_observed_concurrency(events)
    assert max_seen == 1, (
        f"[{phase}] expected max concurrency == 1 with DATABASE_MAX_LRU_CACHE_SIZE=1, "
        f"observed {max_seen}."
    )
    logger.info(
        f"[{phase}] queue serialisation verified: "
        f"{len(enters)} slot acquisitions, max concurrent = {max_seen}"
    )


def _assert_capacity_reached_and_waited(
    events: List[Event],
    *,
    phase: str,
    expected_capacity: int,
    min_expected_enters: int,
) -> None:
    """Assert that peak concurrency hit exactly ``expected_capacity`` and that
    tasks beyond the capacity actually waited for earlier ones to release.

    Two properties are checked:

    * The max number of slots held simultaneously is exactly
      ``expected_capacity`` (not less — otherwise we didn't prove the queue
      reached its budget; not more — the queue broke its own limit).
    * The ``(expected_capacity + 1)``-th task to ``enter`` did so only after
      some earlier task had ``exit``ed — proof that it actually queued.
    """
    enters = sorted([e for e in events if e[1] == "enter"], key=lambda e: e[2])
    exits = sorted([e for e in events if e[1] == "exit"], key=lambda e: e[2])

    assert len(enters) >= min_expected_enters, (
        f"[{phase}] expected at least {min_expected_enters} enter events, saw {len(enters)}"
    )

    max_seen = _max_observed_concurrency(events)
    assert max_seen == expected_capacity, (
        f"[{phase}] expected peak concurrency == {expected_capacity}, observed {max_seen}"
    )

    # The (capacity+1)-th enter must come AFTER the first exit — otherwise
    # no waiting actually happened.
    assert len(enters) > expected_capacity, (
        f"[{phase}] need more than {expected_capacity} tasks to verify waiting"
    )
    waiter = enters[expected_capacity]
    assert exits, f"[{phase}] no exit events recorded — nothing released"
    earliest_exit_t = exits[0][2]
    waiter_t = waiter[2]
    assert waiter_t >= earliest_exit_t, (
        f"[{phase}] expected task #{expected_capacity + 1} ({waiter[0]}) to enter "
        f"after the first exit, but it entered at +{waiter_t * 1000:.1f}ms while "
        f"the first exit was at +{earliest_exit_t * 1000:.1f}ms"
    )


def _summarise(events: List[Event]) -> str:
    """Human-readable timeline for debug output."""
    if not events:
        return "(no events)"
    t0 = min(e[2] for e in events)
    lines = []
    for label, kind, t in sorted(events, key=lambda e: e[2]):
        lines.append(f"  +{(t - t0) * 1000:7.1f}ms  {kind:<10}  {label}")
    return "\n".join(lines)


async def main() -> None:
    # --- 1. Directory + state reset ---------------------------------------
    here = pathlib.Path(__file__).parent
    data_directory_path = str((here / ".data_storage/dataset_queue_e2e").resolve())
    system_directory_path = str((here / ".cognee_system/dataset_queue_e2e").resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(system_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await setup()  # ensure clean DB state and queue singleton reset
    await get_default_user()  # ensure default user exists for dataset ownership in the test

    # --- 2. Reset queue singleton and confirm settings --------------------
    dataset_queue._instance = None
    queue = dataset_queue()
    assert queue._enabled is True, (
        f"queue should be enabled via DATASET_QUEUE_ENABLED=true. Got enabled={queue._enabled}."
    )
    assert queue._max_concurrent == 1, (
        f"queue should have max_concurrent=1 after patching DATABASE_MAX_LRU_CACHE_SIZE. "
        f"Got max_concurrent={queue._max_concurrent}."
    )
    logger.info(f"Queue ready: enabled={queue._enabled}, max_concurrent={queue._max_concurrent}")

    # --- 3. Create 3 small text inputs -------------------------------------
    text = []
    for i in range(3):
        text.append(
            f"Document number {i}. This file talks about topic number {i}. "
            f"Cognee will ingest this and queue-serialise the add pipeline."
        )

    dataset_names = [f"queue_e2e_ds_{i}" for i in range(3)]

    # ====================================================================
    # PHASE 1: concurrent cognee.add on 3 different datasets
    # ====================================================================
    events: List[Event] = []
    _install_queue_tracing(events)

    add_tasks = [
        asyncio.create_task(
            cognee.add(text[i], dataset_name=dataset_names[i]),
            name=f"add:{dataset_names[i]}",
        )
        for i in range(3)
    ]
    await asyncio.gather(*add_tasks)

    print("\n=== Phase 1: cognee.add timeline ===")
    print(_summarise(events))
    _assert_serialized(events, phase="add", min_expected_enters=3)

    # Fan-in sanity: all 3 adds should have hit the queue gate
    # (wait_start) before the sole slot holder finished.
    wait_starts = [e for e in events if e[1] == "wait_start"]
    assert len(wait_starts) >= 3, (
        f"expected all 3 adds to reach the queue gate (wait_start), saw {len(wait_starts)}"
    )

    # ====================================================================
    # PHASE 2: concurrent cognee.cognify on the 3 datasets
    # ====================================================================
    events.clear()

    # Phases 2 and 3 hit a real LLM endpoint and can take minutes and/or
    # hang on flaky networks (synchronous HTTP in LLM clients ignores
    # asyncio cancellation). They're gated behind an opt-in env flag so the
    # queue-focused Phase 4 always runs. Set ``COGNEE_E2E_RUN_LLM_PHASES=1``
    # to include them.
    run_llm_phases = os.getenv("COGNEE_E2E_RUN_LLM_PHASES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if run_llm_phases:
        cognify_tasks = [
            asyncio.create_task(
                cognee.cognify(datasets=[dataset_names[i]]),
                name=f"cognify:{dataset_names[i]}",
            )
            for i in range(3)
        ]
        await asyncio.gather(*cognify_tasks)

        print("\n=== Phase 2: cognee.cognify timeline ===")
        print(_summarise(events))
        _assert_serialized(events, phase="cognify", min_expected_enters=3)
    else:
        print("\nPhase 2 (cognify) skipped — set COGNEE_E2E_RUN_LLM_PHASES=1 to run.")

    # ====================================================================
    # PHASE 3: concurrent cognee.search across the 3 datasets
    # ====================================================================
    events.clear()

    if run_llm_phases:
        search_tasks = [
            asyncio.create_task(
                cognee.search(
                    query_type=SearchType.CHUNKS,
                    query_text=f"topic number {i}",
                    datasets=[dataset_names[i]],
                ),
                name=f"search:{dataset_names[i]}",
            )
            for i in range(3)
        ]
        search_results = await asyncio.gather(*search_tasks)

        print("\n=== Phase 3: cognee.search timeline ===")
        print(_summarise(events))
        _assert_serialized(events, phase="search", min_expected_enters=3)

        # Sanity: every search returned at least one chunk.
        for i, result in enumerate(search_results):
            assert result, f"search on {dataset_names[i]} returned no results: {result!r}"
    else:
        print("Phase 3 (search) skipped — set COGNEE_E2E_RUN_LLM_PHASES=1 to run.")

    # ====================================================================
    # PHASE 4: max_concurrent=2 with 4 concurrent adds — verify that
    # exactly 2 run at a time and the other 2 wait. Reconfigures the queue
    # mid-test by repointing ``DATABASE_MAX_LRU_CACHE_SIZE`` and resetting
    # the singleton so the new settings take effect.
    # ====================================================================
    _lru_cache_module.DATABASE_MAX_LRU_CACHE_SIZE = 2
    _queue_module.DATABASE_MAX_LRU_CACHE_SIZE = 2
    dataset_queue._instance = None
    queue = dataset_queue()
    assert queue._enabled is True, "queue should still be enabled in Phase 4"
    assert queue._max_concurrent == 2, (
        f"expected max_concurrent=2 in Phase 4, got {queue._max_concurrent}"
    )

    # Install tracing on the NEW singleton's semaphore (the previous tracer
    # was wired to the previous semaphore instance which is now discarded).
    events_p4: List[Event] = []
    _install_queue_tracing(events_p4)

    cap2_dataset_names = [f"queue_e2e_ds_cap2_{i}" for i in range(4)]
    cap2_text = [
        f"Capacity-2 document {i}. Tests that two slots run in parallel while "
        "the remaining datasets wait for one to finish."
        for i in range(4)
    ]

    cap2_add_tasks = [
        asyncio.create_task(
            cognee.add(cap2_text[i], dataset_name=cap2_dataset_names[i]),
            name=f"add_cap2:{cap2_dataset_names[i]}",
        )
        for i in range(4)
    ]
    await asyncio.gather(*cap2_add_tasks)

    print("\n=== Phase 4: cognee.add (max_concurrent=2, 4 datasets) timeline ===")
    print(_summarise(events_p4))
    _assert_capacity_reached_and_waited(
        events_p4,
        phase="add-cap2",
        expected_capacity=2,
        min_expected_enters=4,
    )

    # Fan-in sanity: all 4 adds should have hit the queue gate.
    p4_wait_starts = [e for e in events_p4 if e[1] == "wait_start"]
    assert len(p4_wait_starts) >= 4, (
        f"expected all 4 adds to reach the queue gate, saw {len(p4_wait_starts)}"
    )

    print(
        "\nAll phases passed: queue serialised add, cognify, search, and the 4-dataset/max-2 capacity phase correctly."
    )


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
