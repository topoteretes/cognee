from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, Dataset
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_dataset_data(dataset_id: UUID) -> list[Data]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data)
            .join(Data.datasets)
            .filter((Dataset.id == dataset_id))
            .order_by(Data.data_size.desc())
        )

        data = list(result.scalars().all())

        return data
