"""
Task for automatically deleting unused data from the memify pipeline.

This task identifies and removes entire documents that haven't
been accessed by retrievers for a specified period, helping maintain system
efficiency and storage optimization through whole-document removal.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from uuid import UUID
import os
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, DatasetData
from cognee.shared.logging_utils import get_logger
from sqlalchemy import select, or_
import cognee
import sqlalchemy as sa
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph

logger = get_logger(__name__)


async def cleanup_unused_data(
    minutes_threshold: Optional[int], dry_run: bool = True, user_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    Identify and remove unused data from the memify pipeline.

    Parameters
    ----------
    minutes_threshold : int
        Minutes since last access to consider data unused
    dry_run : bool
        If True, only report what would be deleted without actually deleting (default: True)
    user_id : UUID, optional
        Limit cleanup to specific user's data (default: None)

    Returns
    -------
    Dict[str, Any]
        Cleanup results with status, counts, and timestamp
    """
    # Check 1: Environment variable must be enabled
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":
        logger.warning("Cleanup skipped: ENABLE_LAST_ACCESSED is not enabled.")
        return {
            "status": "skipped",
            "reason": "ENABLE_LAST_ACCESSED not enabled",
            "unused_count": 0,
            "deleted_count": {},
            "cleanup_date": datetime.now(timezone.utc).isoformat(),
        }

    # Check 2: Verify tracking has actually been running
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Count records with non-NULL last_accessed
        tracked_count = await session.execute(
            select(sa.func.count(Data.id)).where(Data.last_accessed.isnot(None))
        )
        tracked_records = tracked_count.scalar()

        if tracked_records == 0:
            logger.warning(
                "Cleanup skipped: No records have been tracked yet. "
                "ENABLE_LAST_ACCESSED may have been recently enabled. "
                "Wait for retrievers to update timestamps before running cleanup."
            )
            return {
                "status": "skipped",
                "reason": "No tracked records found - tracking may be newly enabled",
                "unused_count": 0,
                "deleted_count": {},
                "cleanup_date": datetime.now(timezone.utc).isoformat(),
            }

    logger.info(
        "Starting cleanup task",
        minutes_threshold=minutes_threshold,
        dry_run=dry_run,
        user_id=str(user_id) if user_id else None,
    )

    # Calculate cutoff timestamp
    cutoff_date = datetime.now(timezone.utc) - timedelta(minutes=minutes_threshold)

    # Document-level approach (recommended)
    return await _cleanup_via_sql(cutoff_date, dry_run, user_id)


async def _cleanup_via_sql(
    cutoff_date: datetime, dry_run: bool, user_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    SQL-based cleanup: Query Data table for unused documents and use cognee.delete().

    Parameters
    ----------
    cutoff_date : datetime
        Cutoff date for last_accessed filtering
    dry_run : bool
        If True, only report what would be deleted
    user_id : UUID, optional
        Filter by user ID if provided

    Returns
    -------
    Dict[str, Any]
        Cleanup results
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        # Query for Data records with old last_accessed timestamps
        query = (
            select(Data, DatasetData)
            .join(DatasetData, Data.id == DatasetData.data_id)
            .where(or_(Data.last_accessed < cutoff_date, Data.last_accessed.is_(None)))
        )

        if user_id:
            from cognee.modules.data.models import Dataset

            query = query.join(Dataset, DatasetData.dataset_id == Dataset.id).where(
                Dataset.owner_id == user_id
            )

        result = await session.execute(query)
        unused_data = result.all()

    logger.info(f"Found {len(unused_data)} unused documents in SQL")

    if dry_run:
        return {
            "status": "dry_run",
            "unused_count": len(unused_data),
            "deleted_count": {"data_items": 0, "documents": 0},
            "cleanup_date": datetime.now(timezone.utc).isoformat(),
            "preview": {"documents": len(unused_data)},
        }

    # Delete each document using cognee.delete()
    deleted_count = 0
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user() if user_id is None else None

    for data, dataset_data in unused_data:
        try:
            await cognee.delete(
                data_id=data.id,
                dataset_id=dataset_data.dataset_id,
                mode="hard",  # Use hard mode to also remove orphaned entities
                user=user,
            )
            deleted_count += 1
            logger.info(f"Deleted document {data.id} from dataset {dataset_data.dataset_id}")
        except Exception as e:
            logger.error(f"Failed to delete document {data.id}: {e}")

    logger.info("Cleanup completed", deleted_count=deleted_count)

    return {
        "status": "completed",
        "unused_count": len(unused_data),
        "deleted_count": {"data_items": deleted_count, "documents": deleted_count},
        "cleanup_date": datetime.now(timezone.utc).isoformat(),
    }
