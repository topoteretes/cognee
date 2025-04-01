from .get_relational_engine import get_relational_engine


async def create_db_and_tables():
    relational_engine = get_relational_engine()

    await relational_engine.create_database()
