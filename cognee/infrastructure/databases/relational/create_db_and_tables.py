import asyncio

from .get_relational_engine import get_relational_engine

_create_db_lock = asyncio.Lock()


async def create_db_and_tables():
    """
    Make sure the relational database and its tables exist.

    Called on every hot path (``setup()`` before ``add()``, and every
    ``run_pipeline``), so it must be cheap when there is nothing to do: an
    existing schema is left alone with a single table inspection. Only a
    genuinely empty database is built here, by ``create_database()`` running
    the Alembic chain. Pending migrations on an existing database are NOT
    applied here — that is the once-per-process ``run_migrations()``.
    """
    async with _create_db_lock:
        # Imported at call time: the migrations module sits above this layer.
        from cognee.modules.migrations.startup import _relational_schema_exists

        if await _relational_schema_exists():
            return

        relational_engine = get_relational_engine()

        await relational_engine.create_database()
