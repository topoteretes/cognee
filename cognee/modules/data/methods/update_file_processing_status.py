from typing import List
from uuid import UUID
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

logger = get_logger("file_processing_status")


async def update_file_processing_status_batch(
    file_ids: List[UUID], 
    status: FileProcessingStatus
) -> int:
    """Update file processing status in batches."""
    if not file_ids:
        return 0
    
    # Input validation
    if len(file_ids) > 1000:  # Reasonable batch size limit
        raise ValueError("Cannot update more than 1000 files at once")
    
    logger.info(f"Updating {len(file_ids)} files to status: {status.value}")
    
    # Batch database update with transaction
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        try:
            result = await session.execute(
                update(Data)
                .where(Data.id.in_(file_ids))
                .values(processing_status=status)
            )
            await session.commit()
            updated_count = result.rowcount
            
            if updated_count == 0:
                logger.warning(f"No files updated - file IDs may not exist: {file_ids}")
            
            logger.info(f"Successfully updated {updated_count} files to status: {status.value}")
            return updated_count
            
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error updating file status: {e}")
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Unexpected error updating file status: {e}")
            raise 