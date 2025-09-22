import asyncio
from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError, TimeoutError
from cognee.modules.sync.models import SyncOperation, SyncStatus
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.utils.calculate_backoff import calculate_backoff

logger = get_logger("sync.db_operations")


async def _retry_db_operation(operation_func, run_id: str, max_retries: int = 3):
    """
    Retry database operations with exponential backoff for transient failures.

    Args:
        operation_func: Async function to retry
        run_id: Run ID for logging context
        max_retries: Maximum number of retry attempts

    Returns:
        Result of the operation function

    Raises:
        Exception: Re-raises the last exception if all retries fail
    """
    attempt = 0
    last_exception = None

    while attempt < max_retries:
        try:
            return await operation_func()
        except (DisconnectionError, OperationalError, TimeoutError) as e:
            attempt += 1
            last_exception = e

            if attempt >= max_retries:
                logger.error(
                    f"Database operation failed after {max_retries} attempts for run_id {run_id}: {str(e)}"
                )
                break

            backoff_time = calculate_backoff(attempt - 1)  # calculate_backoff is 0-indexed
            logger.warning(
                f"Database operation failed for run_id {run_id}, retrying in {backoff_time:.2f}s (attempt {attempt}/{max_retries}): {str(e)}"
            )
            await asyncio.sleep(backoff_time)

        except Exception as e:
            # Non-transient errors should not be retried
            logger.error(f"Non-retryable database error for run_id {run_id}: {str(e)}")
            raise

    # If we get here, all retries failed
    raise last_exception


async def update_sync_operation(
    run_id: str,
    status: Optional[SyncStatus] = None,
    progress_percentage: Optional[int] = None,
    records_downloaded: Optional[int] = None,
    total_records_to_sync: Optional[int] = None,
    total_records_to_download: Optional[int] = None,
    total_records_to_upload: Optional[int] = None,
    records_uploaded: Optional[int] = None,
    bytes_downloaded: Optional[int] = None,
    bytes_uploaded: Optional[int] = None,
    dataset_sync_hashes: Optional[dict] = None,
    error_message: Optional[str] = None,
    retry_count: Optional[int] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> Optional[SyncOperation]:
    """
    Update a sync operation record with new status/progress information.

    Args:
        run_id: The public run_id of the sync operation to update
        status: New status for the operation
        progress_percentage: Progress percentage (0-100)
        records_downloaded: Number of records downloaded so far
        total_records_to_sync: Total number of records that need to be synced
        total_records_to_download: Total number of records to download from cloud
        total_records_to_upload: Total number of records to upload to cloud
        records_uploaded: Number of records uploaded so far
        bytes_downloaded: Total bytes downloaded from cloud
        bytes_uploaded: Total bytes uploaded to cloud
        dataset_sync_hashes: Dict mapping dataset_id -> {uploaded: [hashes], downloaded: [hashes]}
        error_message: Error message if operation failed
        retry_count: Number of retry attempts
        started_at: When the actual processing started
        completed_at: When the operation completed (success or failure)

    Returns:
        SyncOperation: The updated sync operation record, or None if not found
    """

    async def _perform_update():
        db_engine = get_relational_engine()

        async with db_engine.get_async_session() as session:
            try:
                # Find the sync operation
                query = select(SyncOperation).where(SyncOperation.run_id == run_id)
                result = await session.execute(query)
                sync_operation = result.scalars().first()

                if not sync_operation:
                    logger.warning(f"Sync operation not found for run_id: {run_id}")
                    return None

                # Log what we're updating for debugging
                updates = []
                if status is not None:
                    updates.append(f"status={status.value}")
                if progress_percentage is not None:
                    updates.append(f"progress={progress_percentage}%")
                if records_downloaded is not None:
                    updates.append(f"downloaded={records_downloaded}")
                if records_uploaded is not None:
                    updates.append(f"uploaded={records_uploaded}")
                if total_records_to_sync is not None:
                    updates.append(f"total_sync={total_records_to_sync}")
                if total_records_to_download is not None:
                    updates.append(f"total_download={total_records_to_download}")
                if total_records_to_upload is not None:
                    updates.append(f"total_upload={total_records_to_upload}")

                if updates:
                    logger.debug(f"Updating sync operation {run_id}: {', '.join(updates)}")

                # Update fields that were provided
                if status is not None:
                    sync_operation.status = status

                if progress_percentage is not None:
                    sync_operation.progress_percentage = max(0, min(100, progress_percentage))

                if records_downloaded is not None:
                    sync_operation.records_downloaded = records_downloaded

                if records_uploaded is not None:
                    sync_operation.records_uploaded = records_uploaded

                if total_records_to_sync is not None:
                    sync_operation.total_records_to_sync = total_records_to_sync

                if total_records_to_download is not None:
                    sync_operation.total_records_to_download = total_records_to_download

                if total_records_to_upload is not None:
                    sync_operation.total_records_to_upload = total_records_to_upload

                if bytes_downloaded is not None:
                    sync_operation.bytes_downloaded = bytes_downloaded

                if bytes_uploaded is not None:
                    sync_operation.bytes_uploaded = bytes_uploaded

                if dataset_sync_hashes is not None:
                    sync_operation.dataset_sync_hashes = dataset_sync_hashes

                if error_message is not None:
                    sync_operation.error_message = error_message

                if retry_count is not None:
                    sync_operation.retry_count = retry_count

                if started_at is not None:
                    sync_operation.started_at = started_at

                if completed_at is not None:
                    sync_operation.completed_at = completed_at

                # Auto-set completion timestamp for terminal statuses
                if (
                    status in [SyncStatus.COMPLETED, SyncStatus.FAILED, SyncStatus.CANCELLED]
                    and completed_at is None
                ):
                    sync_operation.completed_at = datetime.now(timezone.utc)

                # Auto-set started timestamp when moving to IN_PROGRESS
                if status == SyncStatus.IN_PROGRESS and sync_operation.started_at is None:
                    sync_operation.started_at = datetime.now(timezone.utc)

                await session.commit()
                await session.refresh(sync_operation)

                logger.debug(f"Successfully updated sync operation {run_id}")
                return sync_operation

            except SQLAlchemyError as e:
                logger.error(
                    f"Database error updating sync operation {run_id}: {str(e)}", exc_info=True
                )
                await session.rollback()
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error updating sync operation {run_id}: {str(e)}", exc_info=True
                )
                await session.rollback()
                raise

    # Use retry logic for the database operation
    return await _retry_db_operation(_perform_update, run_id)


async def mark_sync_started(run_id: str) -> Optional[SyncOperation]:
    """Convenience method to mark a sync operation as started."""
    return await update_sync_operation(
        run_id=run_id, status=SyncStatus.IN_PROGRESS, started_at=datetime.now(timezone.utc)
    )


async def mark_sync_completed(
    run_id: str,
    records_downloaded: int = 0,
    records_uploaded: int = 0,
    bytes_downloaded: int = 0,
    bytes_uploaded: int = 0,
    dataset_sync_hashes: Optional[dict] = None,
) -> Optional[SyncOperation]:
    """Convenience method to mark a sync operation as completed successfully."""
    return await update_sync_operation(
        run_id=run_id,
        status=SyncStatus.COMPLETED,
        progress_percentage=100,
        records_downloaded=records_downloaded,
        records_uploaded=records_uploaded,
        bytes_downloaded=bytes_downloaded,
        bytes_uploaded=bytes_uploaded,
        dataset_sync_hashes=dataset_sync_hashes,
        completed_at=datetime.now(timezone.utc),
    )


async def mark_sync_failed(run_id: str, error_message: str) -> Optional[SyncOperation]:
    """Convenience method to mark a sync operation as failed."""
    return await update_sync_operation(
        run_id=run_id,
        status=SyncStatus.FAILED,
        error_message=error_message,
        completed_at=datetime.now(timezone.utc),
    )
