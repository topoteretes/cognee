import asyncio

from .get_relational_engine import get_relational_engine

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
