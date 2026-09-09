from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, Dataset


async def get_last_added_data(dataset_id: UUID) -> Data | None:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data)
            .join(Data.datasets)
            .where(Dataset.id == dataset_id)
            .order_by(Data.created_at.desc())
            .limit(1)
        )

        data = result.scalar_one_or_none()

        return data
