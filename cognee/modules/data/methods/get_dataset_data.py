from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from cognee.modules.data.models import Data, Dataset
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_dataset_data(dataset_id: UUID) -> list[Data]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data)
            .join(Data.datasets)
            .options(selectinload(Data.datasets))
            .filter((Dataset.id == dataset_id))
            .order_by(Data.data_size.desc())
        )

        data = list(result.scalars().all())

        for item in data:
            _ = item.datasets

        return data
