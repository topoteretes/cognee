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
    # The relational engine built during the migration above is process-global
    # (@lru_cache) and is now bound to the event loop asyncio.run() just closed.
    # Drop the cache so per-test event loops get a fresh engine — otherwise later
    # async tests that share this cache hit "Event loop is closed" (GeneratorExit
    # on Linux/macOS, a hang on Windows).
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()
