from uuid import UUID

from sqlalchemy import select
from sqlalchemy.sql import func

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import DatasetData


async def has_dataset_data(dataset_id: UUID) -> bool:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        count_query = (
            select(func.count())
            .select_from(DatasetData)
            .where(DatasetData.dataset_id == dataset_id)
        )
        count = await session.execute(count_query)

        return count.scalar_one() > 0
