from uuid import UUID, uuid5, NAMESPACE_OID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from cognee.modules.data.models import Dataset

from cognee.api.v1.datasets.datasets import datasets as cognee_datasets
from cognee.modules.users.models import User


async def create_dataset(dataset_name: str, user: User, session: AsyncSession) -> Dataset:
    owner_id = user.id

    dataset = (
        await session.scalars(
            select(Dataset)
            .options(joinedload(Dataset.data))
            .filter(Dataset.name == dataset_name)
            .filter(Dataset.owner_id == owner_id)
        )
    ).first()

    if dataset is None:
        # Dataset id should be generated based on dataset_name and owner_id/user so multiple users can use the same dataset_name
        dataset_id = await cognee_datasets.get_unique_dataset_id(
            dataset_name=dataset_name, user=user
        )
        dataset = Dataset(id=dataset_id, name=dataset_name, data=[])
        dataset.owner_id = owner_id

        session.add(dataset)

        await session.commit()

    return dataset
