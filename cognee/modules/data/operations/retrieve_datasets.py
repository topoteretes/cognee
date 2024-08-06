from sqlalchemy import select
from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import Dataset

async def retrieve_datasets(dataset_names: list[str]) -> list[Dataset]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        datasets = (await session.scalars(
            select(Dataset).filter(Dataset.name.in_(dataset_names))
        )).all()

        return datasets
