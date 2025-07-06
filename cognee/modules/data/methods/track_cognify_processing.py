from typing import List
from cognee.modules.data.models import FileProcessingStatus
from cognee.modules.data.methods import (
    get_datasets_by_name, 
    get_dataset_data,
    update_file_processing_status_batch
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_processing")


async def prepare_files_for_tracking(datasets: List[str], user_id: str) -> List:
    """Prepare file data for processing status tracking."""
    file_data_items = []
    for dataset_name in datasets:
        dataset_results = await get_datasets_by_name([dataset_name], user_id)
        if dataset_results:
            dataset = dataset_results[0]
            dataset_data = await get_dataset_data(dataset.id)
            file_data_items.extend(dataset_data)
    return file_data_items


async def set_files_processing_status(file_data_items: List, status: FileProcessingStatus) -> None:
    """Set processing status for files with validation."""
    if not file_data_items:
        return
    
    file_ids = [data.id for data in file_data_items]
    await update_file_processing_status_batch(file_ids, status)
    logger.info(f"Set {len(file_ids)} files to {status.value} status") 