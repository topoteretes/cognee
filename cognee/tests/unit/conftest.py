"""Create the relational database once, before any unit test runs.

A number of unit directories open the SQLite database directly and none of
them creates it: api/ (test_live_events), modules/agents, modules/agent_memory
and modules/session_lifecycle at the time of writing. In a full-suite run the
first of them to execute created the database and the rest rode on it; the
moment the suite is sharded across jobs, a shard that starts with any of the
others fails with ``sqlite3.OperationalError: unable to open database file``
or ``no such table``. api/test_live_events fails even alone on a clean
checkout. One session-scoped fixture here makes every shard self-sufficient
by construction, instead of one copy per directory that happens to need it.

Cheap: SQLite in .cognee_system, migrations run in well under a second.
Tests that never touch the relational database are unaffected.
"""

import asyncio
import logging
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from cognee.tests.utils.keyless_gate_targets import KEYLESS_GATE_PATCH_TARGETS

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="session")
def _relational_db_for_unit_tests():
    async def _run():
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.run_migrations import run_migrations

        try:
            await run_migrations()
        except Exception:
            logger.debug("Ignoring exception in _relational_db_for_unit_tests._run", exc_info=True)
            db_engine = get_relational_engine()
            await db_engine.create_database()
            await run_migrations()

    asyncio.run(_run())
    # The engine built above is process-global (@lru_cache) and bound to the
    # event loop asyncio.run() just closed. Drop the cache so per-test loops
    # get a fresh engine -- otherwise async tests that share it hit "Event
    # loop is closed" (GeneratorExit on Linux/macOS, a hang on Windows).
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()


@pytest.fixture(autouse=True)
def _operation_origin_starts_as_the_sdk():
    from cognee.modules.operations.origin import ORIGIN_SDK, operation_origin_scope

    with operation_origin_scope(ORIGIN_SDK):
        yield


# How long a test's leftover engine-close tasks get to finish before its loop
# closes. Closes on a loop finish in milliseconds; the bound only matters when
# one is wedged, and then the test's own timeout is the real limit.
PENDING_CLOSE_DRAIN_SECONDS = 30.0


@pytest.fixture(autouse=True)
def _finish_engine_closes_before_the_test_loop_closes(request):
    """Run an asyncio test's leftover engine-close tasks before its loop closes.

    pytest-asyncio 0.21 closes a test's event loop without cancelling the tasks
    still on it. An engine close that a test starts as its last act — a
    ``cache_clear()`` or ``evict()`` in a teardown, the API lifespan's shutdown —
    is scheduled as a task on that loop (``GRAPH_DATABASE_SUBPROCESS_ENABLED=false``,
    as CI runs) and never runs. The engine cache's pending-close registry then
    holds a future nothing will ever resolve, and the next test that creates an
    engine for the same key waits ``PENDING_CLOSE_WAIT_SECONDS`` on it — 300 s,
    the suite's per-test timeout — so pytest-timeout kills the shard at whatever
    test happens to come first. Which test that is depends on how pytest-split
    balanced the shards, which is why it surfaces as a random hang whenever
    tests are added.

    Driving those tasks to completion here, while the loop is still open, keeps
    every asyncio test hermetic in this respect. Sync tests are untouched: the
    loop fixture is requested only for tests marked ``asyncio``.
    """
    loop = None
    if request.node.get_closest_marker("asyncio") is not None:
        loop = request.getfixturevalue("event_loop")
    yield
    if loop is None or loop.is_closed() or loop.is_running():
        return
    try:
        from cognee.infrastructure.databases.utils.closing_lru_cache import (
            _PENDING_CLOSE_TASKS,
        )
    except Exception:
        logger.debug("Engine cache unavailable; nothing to finish", exc_info=True)
        return
    leftovers = [
        task for task in list(_PENDING_CLOSE_TASKS) if not task.done() and task.get_loop() is loop
    ]
    if not leftovers:
        return
    try:
        loop.run_until_complete(asyncio.wait(leftovers, timeout=PENDING_CLOSE_DRAIN_SECONDS))
    except Exception:
        logger.debug("Ignoring exception while finishing engine closes", exc_info=True)


@pytest.fixture(autouse=True)
def _keyless_gates_see_a_usable_llm():
    """Unit tests mock the LLM call itself; the keyless gates must not skip it.

    CI has no key, so tests that mock the downstream LLM call would otherwise
    take the keyless branch. A meta-test keeps this target list synchronized
    with module-level ``llm_available`` imports.
    """
    with ExitStack() as stack:
        for target in KEYLESS_GATE_PATCH_TARGETS:
            stack.enter_context(patch(target, return_value=True))
        yield
