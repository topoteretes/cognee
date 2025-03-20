from uuid import UUID, uuid5, NAMESPACE_OID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from cognee.modules.data.models import Dataset


async def create_dataset(dataset_name: str, owner_id: UUID, session: AsyncSession) -> Dataset:
    dataset = (
        await session.scalars(
            select(Dataset)
            .options(joinedload(Dataset.data))
            .filter(Dataset.name == dataset_name)
            .filter(Dataset.owner_id == owner_id)
        )
    ).first()

    if dataset is None:
        # Dataset id should be generated based on dataset_name and owner_id so multiple users can use the same dataset_name
        dataset = Dataset(
            id=uuid5(NAMESPACE_OID, f"{dataset_name}{str(owner_id)}"), name=dataset_name, data=[]
        )
        dataset.owner_id = owner_id

        session.add(dataset)

        await session.commit()

    return dataset
