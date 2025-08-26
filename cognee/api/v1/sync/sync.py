import asyncio
import pprint
import uuid
from typing import List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel
from cognee.modules.data.models import Data
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.models import User
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.sync.methods import (
    create_sync_operation, 
    update_sync_operation, 
    mark_sync_started,
    mark_sync_completed,
    mark_sync_failed
)
from cognee.modules.sync.models import SyncStatus


class DataEntryContent(BaseModel):
    """Model for individual data entry with content."""
    id: str
    name: str
    mime_type: Optional[str]
    extension: Optional[str]
    raw_data_location: str
    content: bytes
    node_set: Optional[str] = None
    class Config:
        arbitrary_types_allowed = True  # Allow bytes type


class DatasetMetadata(BaseModel):
    """Model for dataset metadata."""
    id: str
    name: str
    owner_id: str
    created_at: str
    updated_at: Optional[str] = None


class DatasetSyncPayload(BaseModel):
    """Model for the complete sync payload sent to cloud."""
    user_id: str
    dataset_metadata: DatasetMetadata
    data_entries: List[DataEntryContent]
    total_entries: int
    total_size: int
    total_tokens: int

    class Config:
        arbitrary_types_allowed = True  # Allow bytes in nested models


class SyncResponse(BaseModel):
    """Response model for sync operations."""
    run_id: str
    status: str  # "started" for immediate response
    dataset_id: str
    dataset_name: str
    message: str
    timestamp: str
    user_id: str


async def sync(
    dataset: Dataset,
    user: User,
) -> SyncResponse:
    """
    Sync local Cognee data to Cognee Cloud.
    
    This function handles synchronization of local datasets, knowledge graphs, and
    processed data to the Cognee Cloud infrastructure. It uploads local data for
    cloud-based processing, backup, and sharing.
    
    Args:
        dataset: Dataset object to sync (permissions already verified)
        user: User object for authentication and permissions
        
    Returns:
        SyncResponse model with immediate response:
            - run_id: Unique identifier for tracking this sync operation
            - status: Always "started" (sync runs in background)
            - dataset_id: ID of the dataset being synced
            - dataset_name: Name of the dataset being synced
            - message: Description of what's happening
            - timestamp: When the sync was initiated
            - user_id: User who initiated the sync
            
    Raises:
        ConnectionError: If Cognee Cloud service is unreachable
        Exception: For other sync-related errors
    """
    if not dataset:
        raise ValueError("Dataset must be provided for sync operation")
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())
    
    # Get current timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    logger.info(f"Starting cloud sync operation {run_id}: dataset {dataset.name} ({dataset.id})")
    
    # Create sync operation record in database
    try:
        await create_sync_operation(
            run_id=run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            user_id=user.id
        )
        logger.info(f"Created sync operation record for {run_id}")
    except Exception as e:
        logger.error(f"Failed to create sync operation record: {str(e)}")
        # Continue without database tracking if record creation fails
    
    # Start the sync operation in the background
    asyncio.create_task(_perform_background_sync(run_id, dataset, user))
    
    # Return immediately with run_id
    return SyncResponse(
        run_id=run_id,
        status="started",
        dataset_id=str(dataset.id),
        dataset_name=dataset.name,
        message=f"Sync operation started in background. Use run_id '{run_id}' to track progress.",
        timestamp=timestamp,
        user_id=str(user.id)
    )


async def _perform_background_sync(run_id: str, dataset: Dataset, user: User) -> None:
    """Perform the actual sync operation in the background."""
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Background sync {run_id}: Starting sync for dataset {dataset.name} ({dataset.id})")
        
        # Mark sync as in progress
        await mark_sync_started(run_id)
        
        # Perform the actual sync operation
        records_processed, bytes_transferred = await _sync_to_cognee_cloud(dataset, user, run_id)
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Background sync {run_id}: Completed successfully. Records: {records_processed}, Bytes: {bytes_transferred}, Duration: {duration}s")
        
        # Mark sync as completed with final stats
        await mark_sync_completed(run_id, records_processed, bytes_transferred)
        
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"Background sync {run_id}: Failed after {duration}s with error: {str(e)}")
        
        # Mark sync as failed with error message
        await mark_sync_failed(run_id, str(e))


async def _sync_to_cognee_cloud(dataset: Dataset, user: User, run_id: str) -> tuple[int, int]:
    """Sync local data to Cognee Cloud."""
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    logger.info(f"Starting sync to Cognee Cloud: dataset {dataset.name} ({dataset.id})")
    
    try:
        # TODO: Implement actual Cognee Cloud sync logic
        # This would involve:
        # 1. Authenticating with Cognee Cloud service
        # 2. Extracting data from local dataset (knowledge graph, vectors, metadata)
        # 3. Compressing data for efficient transfer
        # 4. Uploading to cloud with proper authentication
        # 5. Verifying data integrity after upload
        
        records_processed = await _extract_and_upload_dataset(dataset, user, run_id)
        
        # TODO: Calculate actual bytes transferred from the extracted content
        # For now using estimate, but this should be the actual size of sync payload sent to cloud
        bytes_transferred = records_processed * 2048
        
        logger.info(f"Successfully synced {records_processed} records ({bytes_transferred} bytes) to Cognee Cloud")
        
        return records_processed, bytes_transferred
        
    except Exception as e:
        logger.error(f"Failed to sync to Cognee Cloud: {str(e)}")
        raise ConnectionError(f"Cloud sync failed: {str(e)}")


async def _extract_and_upload_dataset(dataset: Dataset, user: User, run_id: str) -> int:
    """
    Extract local dataset data and upload to Cognee Cloud.
    
    Args:
        dataset: Dataset to extract and sync
        user: User performing the sync
        run_id: Unique identifier for this sync operation (for progress tracking)
        
    Returns:
        int: Number of records successfully processed
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    try:
        logger.info(f"Extracting data from dataset: {dataset.name} ({dataset.id})")
        
        # Step 1: Get all data entries linked to this dataset
        data_entries = await get_dataset_data(dataset.id)
        logger.info(f"Found {len(data_entries)} data entries in dataset")
        
        # Update sync operation with total records count
        await update_sync_operation(
            run_id=run_id,
            total_records=len(data_entries)
        )
        
        # Step 2: Read contents from each data entry's raw_data_location
        extracted_contents: List[DataEntryContent] = []
        
        for data_entry in data_entries:
            try:
                logger.info(f"Reading content from: {data_entry.name} ({data_entry.raw_data_location})")
                content = await _read_data_content(data_entry)
                
                extracted_contents.append(DataEntryContent(
                    id=str(data_entry.id),
                    name=data_entry.name,
                    mime_type=data_entry.mime_type,
                    extension=data_entry.extension,
                    raw_data_location=data_entry.raw_data_location,
                    content=content,
                    node_set=data_entry.node_set
                ))
                
                # Note: Progress tracking happens during actual cloud upload, not local file reading
                
            except Exception as e:
                logger.warning(f"Failed to read content from {data_entry.name}: {str(e)}")
                # Continue with other entries even if one fails
                continue
        
        logger.info(f"Successfully extracted content from {len(extracted_contents)} data entries")
        
        # Step 3: Prepare data for cloud upload
        # - Serialize extracted contents
        # - Compress if enabled
        # - Create manifest with dataset and content metadata
        
        dataset_metadata = DatasetMetadata(
            id=str(dataset.id),
            name=dataset.name,
            owner_id=str(dataset.owner_id),
            created_at=dataset.created_at.isoformat(),
            updated_at=dataset.updated_at.isoformat() if dataset.updated_at else None,
        )
        
        sync_payload = DatasetSyncPayload(
            user_id=str(user.id),
            dataset_metadata=dataset_metadata,
            data_entries=extracted_contents,
            total_entries=len(extracted_contents),
            total_size=sum(len(entry.content) for entry in extracted_contents),
            total_tokens=sum(len(entry.content) for entry in extracted_contents)
        )

        pprint.pprint(sync_payload)
        
        # Step 4: Upload to cloud (placeholder implementation)
        # TODO: Implement actual cloud upload logic
        # - Authenticate with cloud service
        # - Upload sync payload (with progress tracking here)
        # - Update progress: await update_sync_operation(run_id, progress_percentage=50)
        # - Verify upload integrity
        # - Final progress: await update_sync_operation(run_id, progress_percentage=100)
        
        logger.info(f"Prepared sync payload: {sync_payload.total_entries} entries, {sync_payload.total_size} bytes, {sync_payload.total_tokens} tokens")
        
        # For now, just log the payload structure (remove in production)
        logger.debug(f"Sync payload dataset: {sync_payload.dataset_metadata.name} (ID: {sync_payload.dataset_metadata.id})")
        
        # Return actual count of processed records
        records_processed = len(extracted_contents)
        
        # TODO: Also return actual bytes transferred for more accurate reporting
        # actual_bytes_transferred = sync_payload['total_size']  # This would be the real size
        
        logger.info(f"Extracted and prepared {records_processed} records for cloud upload")
        
        return records_processed
        
    except Exception as e:
        logger.error(f"Failed to extract dataset {dataset.name}: {str(e)}")
        raise


async def _read_data_content(data_entry: Data) -> bytes:
    """Read content from a data entry's raw_data_location as bytes."""
    import os
    import aiofiles
    from cognee.shared.logging_utils import get_logger
    
    logger = get_logger()
    
    try:
        # Handle different types of raw_data_location paths
        raw_location = data_entry.raw_data_location
        
        # Check if file exists
        if not os.path.exists(raw_location):
            logger.warning(f"File not found at raw_data_location: {raw_location}")
            return b""
        
        # Read file content as bytes (works for all file types)
        async with aiofiles.open(raw_location, mode='rb') as file:
            content = await file.read()
            
        logger.debug(f"Successfully read {len(content)} bytes from {raw_location}")
        return content
        
    except Exception as e:
        logger.error(f"Error reading content from {data_entry.raw_data_location}: {str(e)}")
        # Return empty bytes instead of failing completely
        return b""
