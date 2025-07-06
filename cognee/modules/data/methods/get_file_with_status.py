from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_file_with_status(file_id: UUID, dataset_id: UUID) -> Optional[Data]:
    """Get a specific file with its processing status in a dataset."""
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        result = await session.execute(
            select(Data)
            .join(Data.datasets)
            .where(and_(
                Data.id == file_id,
                Data.datasets.any(id=dataset_id)
            ))
        )
        file_data = result.scalar_one_or_none()
        return file_data


async def get_dataset_files_with_status(
    dataset_id: UUID,
    status_filter: Optional[FileProcessingStatus] = None,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Data]:
    """Get all files in a dataset with their processing status, optionally filtered."""
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        # Base query to get files in the dataset
        query = (
            select(Data)
            .join(Data.datasets)
            .where(Data.datasets.any(id=dataset_id))
            .offset(offset)
        )
        
        # Add status filter if provided
        if status_filter:
            query = query.where(Data.processing_status == status_filter)
        
        # Add limit if provided
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        files = result.scalars().all()
        return files 