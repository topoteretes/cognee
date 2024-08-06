from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, Dataset
from cognee.infrastructure.databases.relational import get_relational_engine

async def get_dataset_data(dataset_id: UUID = None, dataset_name: str = None):
    if dataset_id is None and dataset_name is None:
        raise ValueError("get_dataset_data: Either dataset_id or dataset_name must be provided.")

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data).join(Data.datasets).filter((Dataset.id == dataset_id) | (Dataset.name == dataset_name))
        )
        data = result.scalars().all()

        return data
