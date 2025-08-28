import asyncio
import uuid
import os
from typing import List, Optional
from datetime import datetime, timezone

import aiohttp
import aiofiles
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


class LocalFileInfo(BaseModel):
    """Model for local file information with hash."""
    id: str
    name: str
    mime_type: Optional[str]
    extension: Optional[str]
    raw_data_location: str
    content_hash: str  # MD5 hash
    file_size: int
    node_set: Optional[str] = None


class CheckMissingHashesRequest(BaseModel):
    """Request model for checking missing hashes in a dataset"""
    hashes: List[str]


class CheckMissingHashesResponse(BaseModel):
    """Response model for missing hashes check"""
    missing: List[str]


class PruneDatasetRequest(BaseModel):
    """Request model for pruning dataset to specific hashes"""
    items: List[str]


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
    
    # Create sync operation record in database (total_records will be set during background sync)
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
    """
    Sync local data to Cognee Cloud using three-step idempotent process:
    1. Extract local files with stored MD5 hashes and check what's missing on cloud
    2. Upload missing files individually 
    3. Prune cloud dataset to match local state
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    logger.info(f"Starting sync to Cognee Cloud: dataset {dataset.name} ({dataset.id})")
    
    try:
        # Get cloud configuration
        cloud_base_url = await _get_cloud_base_url()
        cloud_auth_token = await _get_cloud_auth_token(user)
        
        logger.info(f"Cloud API URL: {cloud_base_url}")
        
        # Step 1: Extract local file info with stored hashes
        local_files = await _extract_local_files_with_hashes(dataset, user, run_id)
        logger.info(f"Found {len(local_files)} local files to sync")
        
        # Update sync operation with total file count
        try:
            await update_sync_operation(run_id, processed_records=0)
        except Exception as e:
            logger.warning(f"Failed to initialize sync progress: {str(e)}")
        
        if not local_files:
            logger.info("No files to sync - dataset is empty")
            return 0, 0
        
        # Step 2: Check what files are missing on cloud
        local_hashes = [f.content_hash for f in local_files]
        missing_hashes = await _check_missing_hashes(
            cloud_base_url, cloud_auth_token, dataset.id, local_hashes, run_id
        )
        logger.info(f"Cloud is missing {len(missing_hashes)} out of {len(local_hashes)} files")
        
        # Update progress
        try:
            await update_sync_operation(run_id, progress_percentage=25)
        except Exception as e:
            logger.warning(f"Failed to update progress: {str(e)}")
        
        # Step 3: Upload missing files
        bytes_uploaded = await _upload_missing_files(
            cloud_base_url, cloud_auth_token, dataset.id, 
            local_files, missing_hashes, run_id
        )
        logger.info(f"Upload complete: {len(missing_hashes)} files, {bytes_uploaded} bytes")
        
        # Update progress
        try:
            await update_sync_operation(run_id, progress_percentage=75)
        except Exception as e:
            logger.warning(f"Failed to update progress: {str(e)}")
        
        # Step 4: Prune cloud dataset to match local state
        await _prune_cloud_dataset(
            cloud_base_url, cloud_auth_token, dataset.id, local_hashes, run_id
        )
        logger.info("Cloud dataset pruned to match local state")
        
        # Final progress
        try:
            await update_sync_operation(run_id, progress_percentage=100)
        except Exception as e:
            logger.warning(f"Failed to update final progress: {str(e)}")
        
        records_processed = len(local_files)
        total_bytes = sum(f.file_size for f in local_files)
        
        logger.info(f"Sync completed successfully: {records_processed} records, {bytes_uploaded} bytes uploaded")
        
        return records_processed, bytes_uploaded
        
    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        raise ConnectionError(f"Cloud sync failed: {str(e)}")


async def _extract_local_files_with_hashes(dataset: Dataset, user: User, run_id: str) -> List[LocalFileInfo]:
    """
    Extract local dataset data with existing MD5 hashes from database.
    
    Args:
        dataset: Dataset to extract files from
        user: User performing the sync
        run_id: Unique identifier for this sync operation
        
    Returns:
        List[LocalFileInfo]: Information about each local file with stored hash
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    try:
        logger.info(f"Extracting files from dataset: {dataset.name} ({dataset.id})")
        
        # Get all data entries linked to this dataset
        data_entries = await get_dataset_data(dataset.id)
        logger.info(f"Found {len(data_entries)} data entries in dataset")
        
        # Process each data entry to get file info and hash
        local_files: List[LocalFileInfo] = []
        skipped_count = 0
        
        for data_entry in data_entries:
            try:
                # Use existing content_hash from database
                content_hash = data_entry.content_hash
                file_size = data_entry.data_size if data_entry.data_size else 0
                
                # Skip entries without content hash (shouldn't happen in normal cases)
                if not content_hash:
                    skipped_count += 1
                    logger.warning(f"Skipping file {data_entry.name}: missing content_hash in database")
                    continue
                    
                if file_size == 0:
                    # Get file size from filesystem if not stored
                    file_size = await _get_file_size(data_entry.raw_data_location)
                
                local_files.append(LocalFileInfo(
                    id=str(data_entry.id),
                    name=data_entry.name,
                    mime_type=data_entry.mime_type,
                    extension=data_entry.extension,
                    raw_data_location=data_entry.raw_data_location,
                    content_hash=content_hash,
                    file_size=file_size,
                    node_set=data_entry.node_set
                ))
                
            except Exception as e:
                skipped_count += 1
                logger.warning(f"Failed to process file {data_entry.name}: {str(e)}")
                # Continue with other entries even if one fails
                continue
        
        logger.info(f"File extraction complete: {len(local_files)} files processed, {skipped_count} skipped")
        return local_files
        
    except Exception as e:
        logger.error(f"Failed to extract files from dataset {dataset.name}: {str(e)}")
        raise


async def _get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        # Strip file:// prefix if present
        if file_path.startswith('file://'):
            actual_path = file_path[7:]  # Remove 'file://' prefix
        else:
            actual_path = file_path
            
        if os.path.exists(actual_path):
            return os.path.getsize(actual_path)
        return 0
    except Exception:
        return 0


async def _get_cloud_base_url() -> str:
    """Get Cognee Cloud API base URL."""
    # TODO: Make this configurable via environment variable or config
    return os.getenv("COGNEE_CLOUD_API_URL", "http://localhost:8001")


async def _get_cloud_auth_token(user: User) -> str:
    """Get authentication token for Cognee Cloud API."""
    # TODO: Implement proper authentication with Cognee Cloud
    # This should get or refresh an API token for the user
    return os.getenv("COGNEE_CLOUD_AUTH_TOKEN", "your-auth-token-here")


async def _check_missing_hashes(
    cloud_base_url: str, auth_token: str, dataset_id: str, 
    local_hashes: List[str], run_id: str
) -> List[str]:
    """
    Step 1: Check which hashes are missing on cloud.
    
    Returns:
        List[str]: MD5 hashes that need to be uploaded
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    url = f"{cloud_base_url}/api/sync/{dataset_id}/need-data"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    payload = CheckMissingHashesRequest(hashes=local_hashes)
    
    logger.info(f"Checking missing hashes on cloud for dataset {dataset_id}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload.dict(), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    missing_response = CheckMissingHashesResponse(**data)
                    logger.info(f"Cloud reports {len(missing_response.missing)} missing files out of {len(local_hashes)} total")
                    return missing_response.missing
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to check missing hashes: Status {response.status} - {error_text}")
                    raise ConnectionError(f"Failed to check missing hashes: {response.status} - {error_text}")
                    
    except Exception as e:
        logger.error(f"Error checking missing hashes: {str(e)}")
        raise ConnectionError(f"Failed to check missing hashes: {str(e)}")


async def _upload_missing_files(
    cloud_base_url: str, auth_token: str, dataset_id: str,
    local_files: List[LocalFileInfo], missing_hashes: List[str], run_id: str
) -> int:
    """
    Step 2: Upload files that are missing on cloud.
    
    Returns:
        int: Total bytes uploaded
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    # Filter local files to only those with missing hashes
    files_to_upload = [f for f in local_files if f.content_hash in missing_hashes]
    
    logger.info(f"Uploading {len(files_to_upload)} missing files to cloud")
    
    if not files_to_upload:
        logger.info("No files need to be uploaded - all files already exist on cloud")
        return 0
    
    total_bytes_uploaded = 0
    uploaded_count = 0
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/octet-stream"
    }
    
    async with aiohttp.ClientSession() as session:
        for file_info in files_to_upload:
            try:
                # Read file content - strip file:// prefix if present
                raw_path = file_info.raw_data_location
                if raw_path.startswith('file://'):
                    file_path = raw_path[7:]  # Remove 'file://' prefix
                else:
                    file_path = raw_path
                
                async with aiofiles.open(file_path, mode='rb') as file:
                    file_content = await file.read()
                
                # Upload file
                url = f"{cloud_base_url}/api/sync/{dataset_id}/{file_info.content_hash}.{file_info.extension or 'bin'}"
                
                async with session.put(url, data=file_content, headers=headers) as response:
                    if response.status in [200, 201]:
                        total_bytes_uploaded += len(file_content)
                        uploaded_count += 1
                        
                        # Update progress periodically
                        if uploaded_count % 10 == 0:
                            progress = 25 + (uploaded_count / len(files_to_upload)) * 50  # 25-75% range
                            await update_sync_operation(run_id, progress_percentage=int(progress))
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to upload {file_info.name}: Status {response.status} - {error_text}")
                        raise ConnectionError(f"Upload failed for {file_info.name}: HTTP {response.status} - {error_text}")
                        
            except Exception as e:
                logger.error(f"Error uploading file {file_info.name}: {str(e)}")
                raise ConnectionError(f"Upload failed for {file_info.name}: {str(e)}")
    
    logger.info(f"All {uploaded_count} files uploaded successfully: {total_bytes_uploaded} bytes")
    return total_bytes_uploaded


async def _prune_cloud_dataset(
    cloud_base_url: str, auth_token: str, dataset_id: str,
    local_hashes: List[str], run_id: str
) -> None:
    """
    Step 3: Prune cloud dataset to match local state.
    """
    from cognee.shared.logging_utils import get_logger
    logger = get_logger()
    
    url = f"{cloud_base_url}/api/sync/{dataset_id}?prune=true"
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    payload = PruneDatasetRequest(items=local_hashes)
    
    logger.info(f"Pruning cloud dataset to match local state")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.put(url, json=payload.dict(), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data.get('message', 'completed')
                    deleted_entries = data.get('deleted_database_entries', 0)
                    deleted_files = data.get('deleted_files_from_storage', 0)
                    
                    logger.info(f"Cloud dataset pruned successfully: {deleted_entries} entries deleted, {deleted_files} files removed")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to prune cloud dataset: Status {response.status} - {error_text}")
                    # Don't raise error for prune failures - sync partially succeeded
                    
    except Exception as e:
        logger.error(f"Error pruning cloud dataset: {str(e)}")
        # Don't raise error for prune failures - sync partially succeeded
