import asyncio

import pytest


@pytest.fixture(autouse=True, scope="session")
def _setup_db():
    async def _run():
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.run_migrations import run_startup_migrations

        try:
            await run_startup_migrations()
        except Exception:
            db_engine = get_relational_engine()
            await db_engine.create_database()
            await run_startup_migrations()

    asyncio.run(_run())
