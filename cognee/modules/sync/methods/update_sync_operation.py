from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import select
from cognee.modules.sync.models import SyncOperation, SyncStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def update_sync_operation(
    run_id: str,
    status: Optional[SyncStatus] = None,
    progress_percentage: Optional[int] = None,
    processed_records: Optional[int] = None,
    bytes_transferred: Optional[int] = None,
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
        processed_records: Number of records processed so far
        bytes_transferred: Total bytes transferred
        error_message: Error message if operation failed
        retry_count: Number of retry attempts
        started_at: When the actual processing started
        completed_at: When the operation completed (success or failure)

    Returns:
        SyncOperation: The updated sync operation record, or None if not found
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Find the sync operation
        query = select(SyncOperation).where(SyncOperation.run_id == run_id)
        result = await session.execute(query)
        sync_operation = result.scalars().first()

        if not sync_operation:
            return None

        # Update fields that were provided
        if status is not None:
            sync_operation.status = status

        if progress_percentage is not None:
            sync_operation.progress_percentage = max(0, min(100, progress_percentage))

        if processed_records is not None:
            sync_operation.processed_records = processed_records

        if bytes_transferred is not None:
            sync_operation.bytes_transferred = bytes_transferred

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

        return sync_operation


async def mark_sync_started(run_id: str) -> Optional[SyncOperation]:
    """Convenience method to mark a sync operation as started."""
    return await update_sync_operation(
        run_id=run_id, status=SyncStatus.IN_PROGRESS, started_at=datetime.now(timezone.utc)
    )


async def mark_sync_completed(
    run_id: str, processed_records: int, bytes_transferred: int
) -> Optional[SyncOperation]:
    """Convenience method to mark a sync operation as completed successfully."""
    return await update_sync_operation(
        run_id=run_id,
        status=SyncStatus.COMPLETED,
        progress_percentage=100,
        processed_records=processed_records,
        bytes_transferred=bytes_transferred,
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
