from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from cognee.modules.data.models import Dataset
from cognee.infrastructure.databases.relational import get_relational_engine

async def create_dataset(dataset_name: str, owner_id: UUID) -> Dataset:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        dataset = (await session.scalars(
            select(Dataset)\
                .options(joinedload(Dataset.data))\
                .filter(Dataset.name == dataset_name)
                .filter(Dataset.owner_id == owner_id)
        )).first()

        if dataset is None:
            dataset = Dataset(
                name = dataset_name,
                data = []
            )
            dataset.owner_id = owner_id

            session.add(dataset)

            await session.commit()

        return dataset
