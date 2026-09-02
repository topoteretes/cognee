"""Create the relational database before this directory's tests run.

These tests open the SQLite database directly and nothing in them creates
it. In a full-suite run an earlier directory's session fixture (``agents/``)
happened to leave the database behind; the moment the suite is sharded
across jobs, a shard without it fails with ``sqlite3.OperationalError:
no such table`` (or ``unable to open database file`` on a clean checkout).
Same fixture as the sibling, so this directory is self-sufficient.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True, scope="session")
def _setup_db():
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
    # loop asyncio.run() just closed; drop it so per-test loops get a fresh one.
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()
