from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, and_
from cognee.modules.data.models import Data, FileProcessingStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_files_by_status(
    dataset_id: UUID, 
    status: FileProcessingStatus,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Data]:
    """Get files by processing status with pagination."""

    if limit is not None and (limit < 1 or limit > 1000):
        raise ValueError("limit must be between 1 and 1000")
    
    if offset < 0:
        raise ValueError("offset must be non-negative")
    
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        query = (
            select(Data)
            .join(Data.datasets)
            .where(and_(
                Data.datasets.any(id=dataset_id),
                Data.processing_status == status
            ))
            .offset(offset)
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await session.execute(query)
        files = result.scalars().all()
        
        return files 