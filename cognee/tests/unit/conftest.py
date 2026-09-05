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

import pytest


@pytest.fixture(autouse=True, scope="session")
def _relational_db_for_unit_tests():
    async def _run():
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.run_migrations import run_migrations

        try:
            await run_migrations()
        except Exception:
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
