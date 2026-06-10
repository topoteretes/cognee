from typing import Union
from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset


async def get_datasets_by_name(
    dataset_names: Union[str, list[str]], user_id: UUID
) -> list[Dataset]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        datasets = (
            await session.scalars(
                select(Dataset)
                .filter(Dataset.owner_id == user_id)
                .filter(Dataset.name.in_(dataset_names))
            )
        ).all()

        return datasets
