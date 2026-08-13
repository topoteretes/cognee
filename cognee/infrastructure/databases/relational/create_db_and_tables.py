import asyncio

from .get_relational_engine import get_relational_engine

# Table creation is driven purely by Base.metadata, so a table exists at runtime
# only if its module was imported. Registered here rather than only in
# cognee/__init__.py so `create_db_and_tables()` also creates them for callers
# that reach the relational package without importing the top-level SDK.
import cognee.modules.recall_coverage.models  # noqa: E402, F401  (recall_coverage_*)

_create_db_lock = asyncio.Lock()


async def create_db_and_tables():
    """
    Create a database and its tables.

    This asynchronous function retrieves the relational engine and calls its method to
    create a database.
    """
    async with _create_db_lock:
        relational_engine = get_relational_engine()

        await relational_engine.create_database()
