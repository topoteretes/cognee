from uuid import UUID
from collections import defaultdict
from sqlalchemy import select
from cognee.modules.data.models import Data, FileProcessingStatus, ProcessingMetrics
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_processing_metrics(dataset_id: UUID) -> ProcessingMetrics:
    """Get processing metrics for a dataset."""

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Get all files in the dataset with their status
        query = (
            select(Data.processing_status)
            .join(Data.datasets)
            .where(Data.datasets.any(id=dataset_id))
        )
        
        result = await session.execute(query)
        statuses = result.scalars().all()
        
        # Count by status
        status_counts = defaultdict(int)
        for status in statuses:
            status = status or FileProcessingStatus.UNPROCESSED
            status_counts[status] += 1
        
        metrics = ProcessingMetrics(
            total_files=sum(status_counts.values()),
            processed_files=status_counts.get(FileProcessingStatus.PROCESSED, 0),
            failed_files=status_counts.get(FileProcessingStatus.ERROR, 0),
            processing_files=status_counts.get(FileProcessingStatus.PROCESSING, 0)
        )
        
        return metrics 