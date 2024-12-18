from cognee.infrastructure.files.storage import LocalStorage
from .ModelBase import Base
from .get_relational_engine import get_relational_engine, get_relational_config


async def create_db_and_tables():
    relational_config = get_relational_config()
    relational_engine = get_relational_engine()

    if relational_engine.engine.dialect.name == "sqlite":
        LocalStorage.ensure_directory_exists(relational_config.db_path)

    async with relational_engine.engine.begin() as connection:
        if len(Base.metadata.tables.keys()) > 0:
            await connection.run_sync(Base.metadata.create_all)
