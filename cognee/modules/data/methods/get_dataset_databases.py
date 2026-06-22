from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import DatasetDatabase


async def get_dataset_databases() -> list[DatasetDatabase]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset_databases = (await session.scalars(select(DatasetDatabase))).all()
        return dataset_databases
