import io
import os
import uuid
import asyncio
import aiohttp
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from cognee.api.v1.cognify import cognify

from cognee.infrastructure.files.storage import get_file_storage
from cognee.tasks.ingestion.ingest_data import ingest_data
from cognee.shared.logging_utils import get_logger
from cognee.modules.users.models import User
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods import get_dataset_data
from cognee.modules.sync.methods import (
    create_sync_operation,
    update_sync_operation,
    mark_sync_started,
    mark_sync_completed,
    mark_sync_failed,
)
from cognee.shared.utils import create_secure_ssl_context

logger = get_logger("sync")


async def _safe_update_progress(run_id: str, stage: str, **kwargs):
    """
    Safely update sync progress with better error handling and context.

    Args:
        run_id: Sync operation run ID
        progress_percentage: Progress percentage (0-100)
        stage: Description of current stage for logging
        **kwargs: Additional fields to update (records_downloaded, records_uploaded, etc.)
    """
    try:
        await update_sync_operation(run_id, **kwargs)
        logger.info(f"Sync {run_id}: Progress updated during {stage}")
    except Exception as e:
        # Log error but don't fail the sync - progress updates are nice-to-have
        logger.warning(
            f"Sync {run_id}: Non-critical progress update failed during {stage}: {str(e)}"
        )
        # Continue without raising - sync operation is more important than progress tracking


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

    dataset_id: str
    dataset_name: str
    hashes: List[str]


class CheckHashesDiffResponse(BaseModel):
    """Response model for missing hashes check"""

    missing_on_remote: List[str]
    missing_on_local: List[str]


class PruneDatasetRequest(BaseModel):
    """Request model for pruning dataset to specific hashes"""

    items: List[str]


class SyncResponse(BaseModel):
    """Response model for sync operations."""

    run_id: str
    status: str  # "started" for immediate response
    dataset_ids: List[str]
    dataset_names: List[str]
    message: str
    timestamp: str
    user_id: str


async def sync(
    datasets: List[Dataset],
    user: User,
) -> SyncResponse:
    """
    Sync local Cognee data to Cognee Cloud.

    This function handles synchronization of multiple datasets, knowledge graphs, and
    processed data to the Cognee Cloud infrastructure. It uploads local data for
    cloud-based processing, backup, and sharing.

    Args:
        datasets: List of Dataset objects to sync (permissions already verified)
        user: User object for authentication and permissions

    Returns:
        SyncResponse model with immediate response:
            - run_id: Unique identifier for tracking this sync operation
            - status: Always "started" (sync runs in background)
            - dataset_ids: List of dataset IDs being synced
            - dataset_names: List of dataset names being synced
            - message: Description of what's happening
            - timestamp: When the sync was initiated
            - user_id: User who initiated the sync

    Raises:
        ConnectionError: If Cognee Cloud service is unreachable
        Exception: For other sync-related errors
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided for sync operation")

    # Generate a unique run ID
    run_id = str(uuid.uuid4())

    # Get current timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    dataset_info = ", ".join([f"{d.name} ({d.id})" for d in datasets])
    logger.info(f"Starting cloud sync operation {run_id}: datasets {dataset_info}")

    # Create sync operation record in database (total_records will be set during background sync)
    try:
        await create_sync_operation(
            run_id=run_id,
            dataset_ids=[d.id for d in datasets],
            dataset_names=[d.name for d in datasets],
            user_id=user.id,
        )
        logger.info(f"Created sync operation record for {run_id}")
    except Exception as e:
        logger.error(f"Failed to create sync operation record: {str(e)}")
        # Continue without database tracking if record creation fails

    # Start the sync operation in the background
    asyncio.create_task(_perform_background_sync(run_id, datasets, user))

    # Return immediately with run_id
    return SyncResponse(
        run_id=run_id,
        status="started",
        dataset_ids=[str(d.id) for d in datasets],
        dataset_names=[d.name for d in datasets],
        message=f"Sync operation started in background for {len(datasets)} datasets. Use run_id '{run_id}' to track progress.",
        timestamp=timestamp,
        user_id=str(user.id),
    )


async def _perform_background_sync(run_id: str, datasets: List[Dataset], user: User) -> None:
    """Perform the actual sync operation in the background for multiple datasets."""
    start_time = datetime.now(timezone.utc)

    try:
        dataset_info = ", ".join([f"{d.name} ({d.id})" for d in datasets])
        logger.info(f"Background sync {run_id}: Starting sync for datasets {dataset_info}")

        # Mark sync as in progress
        await mark_sync_started(run_id)

        # Perform the actual sync operation
        MAX_RETRY_COUNT = 3
        retry_count = 0
        while retry_count < MAX_RETRY_COUNT:
            try:
                (
                    records_downloaded,
                    records_uploaded,
                    bytes_downloaded,
                    bytes_uploaded,
                    dataset_sync_hashes,
                ) = await _sync_to_cognee_cloud(datasets, user, run_id)
                break
            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Background sync {run_id}: Failed after {retry_count} retries with error: {str(e)}"
                )
                await update_sync_operation(run_id, retry_count=retry_count)
                await asyncio.sleep(2**retry_count)
                continue

        if retry_count == MAX_RETRY_COUNT:
            logger.error(f"Background sync {run_id}: Failed after {MAX_RETRY_COUNT} retries")
            await mark_sync_failed(run_id, "Failed after 3 retries")
            return

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        logger.info(
            f"Background sync {run_id}: Completed successfully. Downloaded: {records_downloaded} records/{bytes_downloaded} bytes, Uploaded: {records_uploaded} records/{bytes_uploaded} bytes, Duration: {duration}s"
        )

        # Mark sync as completed with final stats and data lineage
        await mark_sync_completed(
            run_id,
            records_downloaded,
            records_uploaded,
            bytes_downloaded,
            bytes_uploaded,
            dataset_sync_hashes,
        )

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        logger.error(f"Background sync {run_id}: Failed after {duration}s with error: {str(e)}")

        # Mark sync as failed with error message
        await mark_sync_failed(run_id, str(e))


async def _sync_to_cognee_cloud(
    datasets: List[Dataset], user: User, run_id: str
) -> tuple[int, int, int, int, dict]:
    """
    Sync local data to Cognee Cloud using three-step idempotent process:
    1. Extract local files with stored MD5 hashes and check what's missing on cloud
    2. Upload missing files individually
    3. Prune cloud dataset to match local state
    """
    dataset_info = ", ".join([f"{d.name} ({d.id})" for d in datasets])
    logger.info(f"Starting sync to Cognee Cloud: datasets {dataset_info}")

    total_records_downloaded = 0
    total_records_uploaded = 0
    total_bytes_downloaded = 0
    total_bytes_uploaded = 0
    dataset_sync_hashes = {}

    try:
        # Get cloud configuration
        cloud_base_url = await _get_cloud_base_url()
        cloud_auth_token = await _get_cloud_auth_token(user)

        # Step 1: Sync files for all datasets concurrently
        sync_files_tasks = [
            _sync_dataset_files(dataset, cloud_base_url, cloud_auth_token, user, run_id)
            for dataset in datasets
        ]

        logger.info(f"Starting concurrent file sync for {len(datasets)} datasets")

        has_any_uploads = False
        has_any_downloads = False
        processed_datasets = []
        completed_datasets = 0

        # Process datasets concurrently and accumulate results
        for completed_task in asyncio.as_completed(sync_files_tasks):
            try:
                dataset_result = await completed_task
                completed_datasets += 1

                # Update progress based on completed datasets (0-80% for file sync)
                file_sync_progress = int((completed_datasets / len(datasets)) * 80)
                await _safe_update_progress(
                    run_id, "file_sync", progress_percentage=file_sync_progress
                )

                if dataset_result is None:
                    logger.info(
                        f"Progress: {completed_datasets}/{len(datasets)} datasets processed ({file_sync_progress}%)"
                    )
                    continue

                total_records_downloaded += dataset_result.records_downloaded
                total_records_uploaded += dataset_result.records_uploaded
                total_bytes_downloaded += dataset_result.bytes_downloaded
                total_bytes_uploaded += dataset_result.bytes_uploaded

                # Build per-dataset hash tracking for data lineage
                dataset_sync_hashes[dataset_result.dataset_id] = {
                    "uploaded": dataset_result.uploaded_hashes,
                    "downloaded": dataset_result.downloaded_hashes,
                }

                if dataset_result.has_uploads:
                    has_any_uploads = True
                if dataset_result.has_downloads:
                    has_any_downloads = True

                processed_datasets.append(dataset_result.dataset_id)

                logger.info(
                    f"Progress: {completed_datasets}/{len(datasets)} datasets processed ({file_sync_progress}%) - "
                    f"Completed file sync for dataset {dataset_result.dataset_name}: "
                    f"↑{dataset_result.records_uploaded} files ({dataset_result.bytes_uploaded} bytes), "
                    f"↓{dataset_result.records_downloaded} files ({dataset_result.bytes_downloaded} bytes)"
                )
            except Exception as e:
                completed_datasets += 1
                logger.error(f"Dataset file sync failed: {str(e)}")
                # Update progress even for failed datasets
                file_sync_progress = int((completed_datasets / len(datasets)) * 80)
                await _safe_update_progress(
                    run_id, "file_sync", progress_percentage=file_sync_progress
                )
                # Continue with other datasets even if one fails

        # Step 2: Trigger cognify processing once for all datasets (only if any files were uploaded)
        # Update progress to 90% before cognify
        await _safe_update_progress(run_id, "cognify", progress_percentage=90)

        if has_any_uploads and processed_datasets:
            logger.info(
                f"Progress: 90% - Triggering cognify processing for {len(processed_datasets)} datasets with new files"
            )
            try:
                # Trigger cognify for all datasets at once - use first dataset as reference point
                await _trigger_remote_cognify(
                    cloud_base_url, cloud_auth_token, datasets[0].id, run_id
                )
                logger.info("Cognify processing triggered successfully for all datasets")
            except Exception as e:
                logger.warning(f"Failed to trigger cognify processing: {str(e)}")
                # Don't fail the entire sync if cognify fails
        else:
            logger.info(
                "Progress: 90% - Skipping cognify processing - no new files were uploaded across any datasets"
            )

        # Step 3: Trigger local cognify processing if any files were downloaded
        if has_any_downloads and processed_datasets:
            logger.info(
                f"Progress: 95% - Triggering local cognify processing for {len(processed_datasets)} datasets with downloaded files"
            )
            try:
                await cognify()
                logger.info("Local cognify processing completed successfully for all datasets")
            except Exception as e:
                logger.warning(f"Failed to run local cognify processing: {str(e)}")
                # Don't fail the entire sync if local cognify fails
        else:
            logger.info(
                "Progress: 95% - Skipping local cognify processing - no new files were downloaded across any datasets"
            )

        # Update final progress
        try:
            await _safe_update_progress(
                run_id,
                "final",
                progress_percentage=100,
                total_records_to_sync=total_records_uploaded + total_records_downloaded,
                total_records_to_download=total_records_downloaded,
                total_records_to_upload=total_records_uploaded,
                records_downloaded=total_records_downloaded,
                records_uploaded=total_records_uploaded,
            )
        except Exception as e:
            logger.warning(f"Failed to update final sync progress: {str(e)}")

        logger.info(
            f"Multi-dataset sync completed: {len(datasets)} datasets processed, downloaded {total_records_downloaded} records/{total_bytes_downloaded} bytes, uploaded {total_records_uploaded} records/{total_bytes_uploaded} bytes"
        )

        return (
            total_records_downloaded,
            total_records_uploaded,
            total_bytes_downloaded,
            total_bytes_uploaded,
            dataset_sync_hashes,
        )

    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        raise ConnectionError(f"Cloud sync failed: {str(e)}")


@dataclass
class DatasetSyncResult:
    """Result of syncing files for a single dataset."""

    dataset_name: str
    dataset_id: str
    records_downloaded: int
    records_uploaded: int
    bytes_downloaded: int
    bytes_uploaded: int
    has_uploads: bool  # Whether any files were uploaded (for cognify decision)
    has_downloads: bool  # Whether any files were downloaded (for cognify decision)
    uploaded_hashes: List[str]  # Content hashes of files uploaded during sync
    downloaded_hashes: List[str]  # Content hashes of files downloaded during sync


async def _sync_dataset_files(
    dataset: Dataset, cloud_base_url: str, cloud_auth_token: str, user: User, run_id: str
) -> Optional[DatasetSyncResult]:
    """
    Sync files for a single dataset (2-way: upload to cloud, download from cloud).
    Does NOT trigger cognify - that's done separately once for all datasets.

    Returns:
        DatasetSyncResult with sync results or None if dataset was empty
    """
    logger.info(f"Syncing files for dataset: {dataset.name} ({dataset.id})")

    try:
        # Step 1: Extract local file info with stored hashes
        local_files = await _extract_local_files_with_hashes(dataset, user, run_id)
        logger.info(f"Found {len(local_files)} local files for dataset {dataset.name}")

        if not local_files:
            logger.info(f"No files to sync for dataset {dataset.name} - skipping")
            return None

        # Step 2: Check what files are missing on cloud
        local_hashes = [f.content_hash for f in local_files]
        hashes_diff_response = await _check_hashes_diff(
            cloud_base_url, cloud_auth_token, dataset, local_hashes, run_id
        )

        hashes_missing_on_remote = hashes_diff_response.missing_on_remote
        hashes_missing_on_local = hashes_diff_response.missing_on_local

        logger.info(
            f"Dataset {dataset.name}: {len(hashes_missing_on_remote)} files to upload, {len(hashes_missing_on_local)} files to download"
        )

        # Step 3: Upload files that are missing on cloud
        bytes_uploaded = await _upload_missing_files(
            cloud_base_url, cloud_auth_token, dataset, local_files, hashes_missing_on_remote, run_id
        )
        logger.info(
            f"Dataset {dataset.name}: Upload complete - {len(hashes_missing_on_remote)} files, {bytes_uploaded} bytes"
        )

        # Step 4: Download files that are missing locally
        bytes_downloaded = await _download_missing_files(
            cloud_base_url, cloud_auth_token, dataset, hashes_missing_on_local, user
        )
        logger.info(
            f"Dataset {dataset.name}: Download complete - {len(hashes_missing_on_local)} files, {bytes_downloaded} bytes"
        )

        return DatasetSyncResult(
            dataset_name=dataset.name,
            dataset_id=str(dataset.id),
            records_downloaded=len(hashes_missing_on_local),
            records_uploaded=len(hashes_missing_on_remote),
            bytes_downloaded=bytes_downloaded,
            bytes_uploaded=bytes_uploaded,
            has_uploads=len(hashes_missing_on_remote) > 0,
            has_downloads=len(hashes_missing_on_local) > 0,
            uploaded_hashes=hashes_missing_on_remote,
            downloaded_hashes=hashes_missing_on_local,
        )

    except Exception as e:
        logger.error(f"Failed to sync files for dataset {dataset.name} ({dataset.id}): {str(e)}")
        raise  # Re-raise to be handled by the caller


async def _extract_local_files_with_hashes(
    dataset: Dataset, user: User, run_id: str
) -> List[LocalFileInfo]:
    """
    Extract local dataset data with existing MD5 hashes from database.

    Args:
        dataset: Dataset to extract files from
        user: User performing the sync
        run_id: Unique identifier for this sync operation

    Returns:
        List[LocalFileInfo]: Information about each local file with stored hash
    """
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
                content_hash = data_entry.raw_content_hash
                file_size = data_entry.data_size if data_entry.data_size else 0

                # Skip entries without content hash (shouldn't happen in normal cases)
                if not content_hash:
                    skipped_count += 1
                    logger.warning(
                        f"Skipping file {data_entry.name}: missing content_hash in database"
                    )
                    continue

                if file_size == 0:
                    # Get file size from filesystem if not stored
                    file_size = await _get_file_size(data_entry.raw_data_location)

                local_files.append(
                    LocalFileInfo(
                        id=str(data_entry.id),
                        name=data_entry.name,
                        mime_type=data_entry.mime_type,
                        extension=data_entry.extension,
                        raw_data_location=data_entry.raw_data_location,
                        content_hash=content_hash,
                        file_size=file_size,
                        node_set=data_entry.node_set,
                    )
                )

            except Exception as e:
                skipped_count += 1
                logger.warning(f"Failed to process file {data_entry.name}: {str(e)}")
                # Continue with other entries even if one fails
                continue

        logger.info(
            f"File extraction complete: {len(local_files)} files processed, {skipped_count} skipped"
        )
        return local_files

    except Exception as e:
        logger.error(f"Failed to extract files from dataset {dataset.name}: {str(e)}")
        raise


async def _get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    try:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_storage = get_file_storage(file_dir)

        return await file_storage.get_size(file_name)
    except Exception:
        return 0


async def _get_cloud_base_url() -> str:
    """Get Cognee Cloud API base URL."""
    return os.getenv("COGNEE_CLOUD_API_URL", "http://localhost:8001")


async def _get_cloud_auth_token(user: User) -> str:
    """Get authentication token for Cognee Cloud API."""
    return os.getenv("COGNEE_CLOUD_AUTH_TOKEN", "your-auth-token")


async def _check_hashes_diff(
    cloud_base_url: str, auth_token: str, dataset: Dataset, local_hashes: List[str], run_id: str
) -> CheckHashesDiffResponse:
    """
    Check which hashes are missing on cloud.

    Returns:
        List[str]: MD5 hashes that need to be uploaded
    """
    url = f"{cloud_base_url}/api/sync/{dataset.id}/diff"
    headers = {"X-Api-Key": auth_token, "Content-Type": "application/json"}

    payload = CheckMissingHashesRequest(
        dataset_id=str(dataset.id), dataset_name=dataset.name, hashes=local_hashes
    )

    logger.info(f"Checking missing hashes on cloud for dataset {dataset.id}")

    try:
        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload.dict(), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    missing_response = CheckHashesDiffResponse(**data)
                    logger.info(
                        f"Cloud is missing {len(missing_response.missing_on_remote)} out of {len(local_hashes)} files, local is missing {len(missing_response.missing_on_local)} files"
                    )
                    return missing_response
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to check missing hashes: Status {response.status} - {error_text}"
                    )
                    raise ConnectionError(
                        f"Failed to check missing hashes: {response.status} - {error_text}"
                    )

    except Exception as e:
        logger.error(f"Error checking missing hashes: {str(e)}")
        raise ConnectionError(f"Failed to check missing hashes: {str(e)}")


async def _download_missing_files(
    cloud_base_url: str,
    auth_token: str,
    dataset: Dataset,
    hashes_missing_on_local: List[str],
    user: User,
) -> int:
    """
    Download files that are missing locally from the cloud.

    Returns:
        int: Total bytes downloaded
    """
    logger.info(f"Downloading {len(hashes_missing_on_local)} missing files from cloud")

    if not hashes_missing_on_local:
        logger.info("No files need to be downloaded - all files already exist locally")
        return 0

    total_bytes_downloaded = 0
    downloaded_count = 0

    headers = {"X-Api-Key": auth_token}

    ssl_context = create_secure_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        for file_hash in hashes_missing_on_local:
            try:
                # Download file from cloud by hash
                download_url = f"{cloud_base_url}/api/sync/{dataset.id}/data/{file_hash}"

                logger.debug(f"Downloading file with hash: {file_hash}")

                async with session.get(download_url, headers=headers) as response:
                    if response.status == 200:
                        file_content = await response.read()
                        file_size = len(file_content)

                        # Get file metadata from response headers
                        file_name = response.headers.get("X-File-Name", f"file_{file_hash}")

                        # Save file locally using ingestion pipeline
                        await _save_downloaded_file(
                            dataset, file_hash, file_name, file_content, user
                        )

                        total_bytes_downloaded += file_size
                        downloaded_count += 1

                        logger.debug(f"Successfully downloaded {file_name} ({file_size} bytes)")

                    elif response.status == 404:
                        logger.warning(f"File with hash {file_hash} not found on cloud")
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to download file {file_hash}: Status {response.status} - {error_text}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error downloading file {file_hash}: {str(e)}")
                continue

    logger.info(
        f"Download summary: {downloaded_count}/{len(hashes_missing_on_local)} files downloaded, {total_bytes_downloaded} bytes total"
    )
    return total_bytes_downloaded


class InMemoryDownload:
    def __init__(self, data: bytes, filename: str):
        self.file = io.BufferedReader(io.BytesIO(data))
        self.filename = filename


async def _save_downloaded_file(
    dataset: Dataset,
    file_hash: str,
    file_name: str,
    file_content: bytes,
    user: User,
) -> None:
    """
    Save a downloaded file to local storage and register it in the dataset.
    Uses the existing ingest_data function for consistency with normal ingestion.

    Args:
        dataset: The dataset to add the file to
        file_hash: MD5 hash of the file content
        file_name: Original file name
        file_content: Raw file content bytes
    """
    try:
        # Create a temporary file-like object from the bytes
        file_obj = InMemoryDownload(file_content, file_name)

        # User is injected as dependency

        # Use the existing ingest_data function to properly handle the file
        # This ensures consistency with normal file ingestion
        await ingest_data(
            data=file_obj,
            dataset_name=dataset.name,
            user=user,
            dataset_id=dataset.id,
        )

        logger.debug(f"Successfully saved downloaded file: {file_name} (hash: {file_hash})")

    except Exception as e:
        logger.error(f"Failed to save downloaded file {file_name}: {str(e)}")
        raise


async def _upload_missing_files(
    cloud_base_url: str,
    auth_token: str,
    dataset: Dataset,
    local_files: List[LocalFileInfo],
    hashes_missing_on_remote: List[str],
    run_id: str,
) -> int:
    """
    Upload files that are missing on cloud.

    Returns:
        int: Total bytes uploaded
    """
    # Filter local files to only those with missing hashes
    files_to_upload = [f for f in local_files if f.content_hash in hashes_missing_on_remote]

    logger.info(f"Uploading {len(files_to_upload)} missing files to cloud")

    if not files_to_upload:
        logger.info("No files need to be uploaded - all files already exist on cloud")
        return 0

    total_bytes_uploaded = 0
    uploaded_count = 0

    headers = {"X-Api-Key": auth_token}

    ssl_context = create_secure_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        for file_info in files_to_upload:
            try:
                file_dir = os.path.dirname(file_info.raw_data_location)
                file_name = os.path.basename(file_info.raw_data_location)
                file_storage = get_file_storage(file_dir)

                async with file_storage.open(file_name, mode="rb") as file:
                    file_content = file.read()

                # Upload file
                url = f"{cloud_base_url}/api/sync/{dataset.id}/data/{file_info.id}"

                request_data = aiohttp.FormData()

                request_data.add_field(
                    "file", file_content, content_type=file_info.mime_type, filename=file_info.name
                )
                request_data.add_field("dataset_id", str(dataset.id))
                request_data.add_field("dataset_name", dataset.name)
                request_data.add_field("data_id", str(file_info.id))
                request_data.add_field("mime_type", file_info.mime_type)
                request_data.add_field("extension", file_info.extension)
                request_data.add_field("md5", file_info.content_hash)

                async with session.put(url, data=request_data, headers=headers) as response:
                    if response.status in [200, 201]:
                        total_bytes_uploaded += len(file_content)
                        uploaded_count += 1
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to upload {file_info.name}: Status {response.status} - {error_text}"
                        )
                        raise ConnectionError(
                            f"Upload failed for {file_info.name}: HTTP {response.status} - {error_text}"
                        )

            except Exception as e:
                logger.error(f"Error uploading file {file_info.name}: {str(e)}")
                raise ConnectionError(f"Upload failed for {file_info.name}: {str(e)}")

    logger.info(f"All {uploaded_count} files uploaded successfully: {total_bytes_uploaded} bytes")
    return total_bytes_uploaded


async def _prune_cloud_dataset(
    cloud_base_url: str, auth_token: str, dataset_id: str, local_hashes: List[str], run_id: str
) -> None:
    """
    Prune cloud dataset to match local state.
    """
    url = f"{cloud_base_url}/api/sync/{dataset_id}?prune=true"
    headers = {"X-Api-Key": auth_token, "Content-Type": "application/json"}

    payload = PruneDatasetRequest(items=local_hashes)

    logger.info("Pruning cloud dataset to match local state")

    try:
        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.put(url, json=payload.dict(), headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    deleted_entries = data.get("deleted_database_entries", 0)
                    deleted_files = data.get("deleted_files_from_storage", 0)

                    logger.info(
                        f"Cloud dataset pruned successfully: {deleted_entries} entries deleted, {deleted_files} files removed"
                    )
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to prune cloud dataset: Status {response.status} - {error_text}"
                    )
                    # Don't raise error for prune failures - sync partially succeeded

    except Exception as e:
        logger.error(f"Error pruning cloud dataset: {str(e)}")
        # Don't raise error for prune failures - sync partially succeeded


async def _trigger_remote_cognify(
    cloud_base_url: str, auth_token: str, dataset_id: str, run_id: str
) -> None:
    """
    Trigger cognify processing on the cloud dataset.

    This initiates knowledge graph processing on the synchronized dataset
    using the cloud infrastructure.
    """
    url = f"{cloud_base_url}/api/cognify"
    headers = {"X-Api-Key": auth_token, "Content-Type": "application/json"}

    payload = {
        "dataset_ids": [str(dataset_id)],  # Convert UUID to string for JSON serialization
        "run_in_background": False,
        "custom_prompt": "",
    }

    logger.info(f"Triggering cognify processing for dataset {dataset_id}")

    try:
        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Cognify processing started successfully: {data}")

                    # Extract pipeline run IDs for monitoring if available
                    if isinstance(data, dict):
                        for dataset_key, run_info in data.items():
                            if isinstance(run_info, dict) and "pipeline_run_id" in run_info:
                                logger.info(
                                    f"Cognify pipeline run ID for dataset {dataset_key}: {run_info['pipeline_run_id']}"
                                )
                else:
                    error_text = await response.text()
                    logger.warning(
                        f"Failed to trigger cognify processing: Status {response.status} - {error_text}"
                    )
                    # TODO: consider adding retries

    except Exception as e:
        logger.warning(f"Error triggering cognify processing: {str(e)}")
        # TODO: consider adding retries
