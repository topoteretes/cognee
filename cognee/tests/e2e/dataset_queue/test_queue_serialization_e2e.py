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
    """Wrap the singleton's ``acquire`` to record enter/exit events.

    Replaces ``acquire`` on the current :func:`dataset_queue` instance so the
    production call sites (``_search_in_dataset_context`` and
    ``run_pipeline_per_dataset``) emit events without code changes.
    """
    queue = dataset_queue()
    original_acquire = queue.acquire

    @asynccontextmanager
    async def _traced_acquire():
        task = asyncio.current_task()
        label = task.get_name() if task is not None else "<detached>"
        events.append((label, "wait_start", time.monotonic()))
        async with original_acquire():
            events.append((label, "enter", time.monotonic()))
            try:
                yield
            finally:
                events.append((label, "exit", time.monotonic()))

    # Replace the acquire method on the singleton instance to capture all calls.
    queue.acquire = _traced_acquire


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

    # ====================================================================
    # PHASE 3: concurrent cognee.search across the 3 datasets
    # ====================================================================
    events.clear()

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

    print("\nAll phases passed: queue serialised add, cognify, and search correctly.")


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
