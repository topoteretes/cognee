from typing import Optional
from uuid import UUID
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset


async def get_dataset(user_id: UUID, dataset_id: UUID) -> Optional[Dataset]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = await session.get(Dataset, dataset_id)

        if dataset and dataset.owner_id != user_id:
            return None

        return dataset
