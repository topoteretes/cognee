from uuid import UUID
from sqlalchemy import select
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_file_processing_status(file_id: UUID) -> FileProcessingStatus:
    """Get file processing status."""
    
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data.processing_status).where(Data.id == file_id)
        )
        status = result.scalar_one_or_none()
        
        if status is None:
            return FileProcessingStatus.UNPROCESSED
        
        return status 