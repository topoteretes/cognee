from typing import List, Dict, Any
from uuid import UUID
from cognee.modules.data.models import FileProcessingStatus
from cognee.modules.data.methods.update_file_processing_status import update_file_processing_status_batch
from cognee.shared.logging_utils import get_logger

logger = get_logger("file_processing_status")


async def reset_file_processing_status(
    file_ids: List[UUID],
    target_status: FileProcessingStatus = FileProcessingStatus.UNPROCESSED
) -> Dict[str, Any]:
    """Reset file processing status."""
    if not file_ids:
        return {"reset_count": 0, "errors": []}
    
    # Input validation

    if len(file_ids) > 100:
        return {"reset_count": 0, "errors": ["Cannot reset more than 100 files at once"]}
    
    try:
        reset_count = await update_file_processing_status_batch(file_ids, target_status)
        logger.info(f"Successfully reset {reset_count} files to {target_status.value}")
        
        return {
            "reset_count": reset_count,
            "errors": [],
            "target_status": target_status.value
        }
    except Exception as e:
        error_msg = f"Failed to reset file processing status: {str(e)}"
        logger.error(error_msg)
        return {
            "reset_count": 0,
            "errors": [error_msg],
            "target_status": target_status.value
        } 