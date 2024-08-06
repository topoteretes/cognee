from .ModelBase import Base
from .get_relational_engine import get_relational_engine

async def create_db_and_tables():
    relational_engine = get_relational_engine()

    async with relational_engine.engine.begin() as connection:
        if len(Base.metadata.tables.keys()) > 0:
            await connection.run_sync(Base.metadata.create_all)
