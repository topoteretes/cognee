from uuid import UUID
from typing import List, Optional
from sqlalchemy import select, desc, and_
from cognee.modules.sync.models import SyncOperation, SyncStatus
from cognee.infrastructure.databases.relational import get_relational_engine


async def get_sync_operation(run_id: str) -> Optional[SyncOperation]:
    """
    Get a sync operation by its run_id.

    Args:
        run_id: The public run_id of the sync operation

    Returns:
        SyncOperation: The sync operation record, or None if not found
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = select(SyncOperation).where(SyncOperation.run_id == run_id)
        result = await session.execute(query)
        return result.scalars().first()


async def get_user_sync_operations(
    user_id: UUID, limit: int = 50, offset: int = 0
) -> List[SyncOperation]:
    """
    Get sync operations for a specific user, ordered by most recent first.

    Args:
        user_id: UUID of the user
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        List[SyncOperation]: List of sync operations for the user
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(SyncOperation)
            .where(SyncOperation.user_id == user_id)
            .order_by(desc(SyncOperation.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(query)
        return list(result.scalars().all())


async def get_sync_operations_by_dataset(
    dataset_id: UUID, limit: int = 50, offset: int = 0
) -> List[SyncOperation]:
    """
    Get sync operations for a specific dataset.

    Args:
        dataset_id: UUID of the dataset
        limit: Maximum number of records to return
        offset: Number of records to skip

    Returns:
        List[SyncOperation]: List of sync operations for the dataset
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(SyncOperation)
            .where(SyncOperation.dataset_id == dataset_id)
            .order_by(desc(SyncOperation.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(query)
        return list(result.scalars().all())


async def get_running_sync_operations_for_user(user_id: UUID) -> List[SyncOperation]:
    """
    Get all currently running sync operations for a specific user.
    Checks for operations with STARTED or IN_PROGRESS status.

    Args:
        user_id: UUID of the user

    Returns:
        List[SyncOperation]: List of running sync operations for the user
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(SyncOperation)
            .where(
                and_(
                    SyncOperation.user_id == user_id,
                    SyncOperation.status.in_([SyncStatus.STARTED, SyncStatus.IN_PROGRESS]),
                )
            )
            .order_by(desc(SyncOperation.created_at))
        )
        result = await session.execute(query)
        return list(result.scalars().all())
