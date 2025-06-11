from uuid import UUID
from cognee.modules.data.models import FileProcessingStatus
from .update_data_processing_status import update_data_processing_status


async def reset_data_processing_status(data_id: UUID) -> None:
    """Reset the processing status of a data record to UNPROCESSED."""
    await update_data_processing_status(data_id, FileProcessingStatus.UNPROCESSED)