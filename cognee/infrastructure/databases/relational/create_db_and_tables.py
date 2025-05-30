from .get_relational_engine import get_relational_engine


async def create_db_and_tables():
    """
    Create a database and its tables.

    This asynchronous function retrieves the relational engine and calls its method to
    create a database.
    """
    relational_engine = get_relational_engine()

    await relational_engine.create_database()
