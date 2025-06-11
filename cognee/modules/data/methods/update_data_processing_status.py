from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine

async def update_data_processing_status(data_id: UUID, new_status: FileProcessingStatus) -> None:
    """Update the processing status of a data record in the database."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Fetch the data record
        data_record = await session.execute(select(Data).filter(Data.id == data_id)).scalar_one_or_none()

        if not data_record:
            raise ValueError(f"Data with id {data_id} not found.")

        # Update the processing status
        data_record.processing_status = new_status

        # Commit the changes
        await session.commit()