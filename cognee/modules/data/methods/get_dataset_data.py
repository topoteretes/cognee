from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, Dataset, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_dataset_data(dataset_id: UUID, statuses: list[FileProcessingStatus] | None = None) -> list[Data]:
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = select(Data).join(Data.datasets).filter(Dataset.id == dataset_id)
        if statuses:
            query = query.filter(Data.processing_status.in_(statuses))
        result = await session.execute(query)

        data = result.scalars().all()

        return data
