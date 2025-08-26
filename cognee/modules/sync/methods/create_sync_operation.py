from uuid import UUID
from typing import Optional
from datetime import datetime, timezone
from cognee.modules.sync.models import SyncOperation, SyncStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def create_sync_operation(
    run_id: str,
    dataset_id: UUID,
    dataset_name: str,
    user_id: UUID,
    total_records: Optional[int] = None
) -> SyncOperation:
    """
    Create a new sync operation record in the database.
    
    Args:
        run_id: Unique public identifier for this sync operation
        dataset_id: UUID of the dataset being synced
        dataset_name: Name of the dataset being synced  
        user_id: UUID of the user who initiated the sync
        total_records: Total number of records to sync (if known)
        
    Returns:
        SyncOperation: The created sync operation record
    """
    db_engine = get_relational_engine()
    
    sync_operation = SyncOperation(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        user_id=user_id,
        status=SyncStatus.STARTED,
        total_records=total_records,
        created_at=datetime.now(timezone.utc)
    )
    
    async with db_engine.get_async_session() as session:
        session.add(sync_operation)
        await session.commit()
        await session.refresh(sync_operation)
        
    return sync_operation
