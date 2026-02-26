from .get_relational_engine import get_relational_engine


async def create_db_and_tables():
    """
    Ensure the database exists and is reachable.

    For SQLite this creates the database file and parent directory if needed.
    Schema creation is handled exclusively by Alembic migrations.
    """
    relational_engine = get_relational_engine()

    await relational_engine.create_database()
