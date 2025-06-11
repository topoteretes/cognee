from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine

async def update_data_processing_status(data_id: UUID, new_status: FileProcessingStatus) -> None:
    """Update the processing status of a data record in the database."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        result = await session.execute(select(Data).filter(Data.id == data_id))
        data_record = result.scalar_one_or_none()

        if not data_record:
            raise ValueError(f"Data with id {data_id} not found.")

        data_record.processing_status = new_status

        await session.commit()