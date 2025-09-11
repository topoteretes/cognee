from uuid import UUID
from typing import Optional, List
from datetime import datetime, timezone
from cognee.modules.sync.models import SyncOperation, SyncStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def create_sync_operation(
    run_id: str,
    dataset_ids: List[UUID],
    dataset_names: List[str],
    user_id: UUID,
    total_records_to_sync: Optional[int] = None,
    total_records_to_download: Optional[int] = None,
    total_records_to_upload: Optional[int] = None,
) -> SyncOperation:
    """
    Create a new sync operation record in the database.

    Args:
        run_id: Unique public identifier for this sync operation
        dataset_ids: List of dataset UUIDs being synced
        dataset_names: List of dataset names being synced
        user_id: UUID of the user who initiated the sync
        total_records_to_sync: Total number of records to sync (if known)
        total_records_to_download: Total number of records to download (if known)
        total_records_to_upload: Total number of records to upload (if known)

    Returns:
        SyncOperation: The created sync operation record
    """
    db_engine = get_relational_engine()

    sync_operation = SyncOperation(
        run_id=run_id,
        dataset_ids=[
            str(uuid) for uuid in dataset_ids
        ],  # Convert UUIDs to strings for JSON storage
        dataset_names=dataset_names,
        user_id=user_id,
        status=SyncStatus.STARTED,
        total_records_to_sync=total_records_to_sync,
        total_records_to_download=total_records_to_download,
        total_records_to_upload=total_records_to_upload,
        created_at=datetime.now(timezone.utc),
    )

    async with db_engine.get_async_session() as session:
        session.add(sync_operation)
        await session.commit()
        await session.refresh(sync_operation)

    return sync_operation
