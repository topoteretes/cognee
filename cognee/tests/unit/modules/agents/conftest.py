import pytest


@pytest.fixture(autouse=True, scope="session")
async def _setup_db():
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.run_migrations import run_startup_migrations

    try:
        await run_startup_migrations()
    except Exception:
        db_engine = get_relational_engine()
        await db_engine.create_database()
        await run_startup_migrations()
