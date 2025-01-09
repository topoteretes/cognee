from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset


async def get_datasets(user_id: UUID) -> list[Dataset]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        datasets = (
            await session.scalars(select(Dataset).filter(Dataset.owner_id == user_id))
        ).all()

        return datasets
