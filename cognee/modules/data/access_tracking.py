"""Data-level access tracking utilities.

This module provides functions to track when Data entries are accessed,
enabling efficient cleanup of unused data. By working at the Data level,
it ensures proper cleanup of related graph and vector database entries.

The data_access_tracking reference table approach avoids frequent writes
on the main Data table while maintaining efficient access tracking.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, insert, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def track_data_access(
    session: AsyncSession,
    data_id: UUID,
) -> bool:
    """
    Track access to a Data entry by updating/creating record in data_access_tracking.
    
    Uses PostgreSQL's ON CONFLICT to efficiently update existing records or insert new ones.
    This approach handles concurrent access and avoids race conditions.
    
    Args:
        session: Database session
        data_id: ID of the Data entry being accessed
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> await track_data_access(session, data_id=uuid_obj)
    """
    try:
        # Use PostgreSQL's INSERT ... ON CONFLICT for efficient upsert
        stmt = pg_insert(
            'data_access_tracking'
        ).values(
            data_id=data_id,
            last_accessed=datetime.now(timezone.utc),
            access_count=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        ).on_conflict_do_update(
            index_elements=['data_id'],
            set_={
                'last_accessed': datetime.now(timezone.utc),
                'access_count': 'data_access_tracking.access_count + 1',
                'updated_at': datetime.now(timezone.utc)
            }
        )
        
        await session.execute(stmt)
        return True
        
    except Exception as e:
        logger.error(f"Failed to track access for Data {data_id}: {e}")
        return False


async def bulk_track_data_access(
    session: AsyncSession,
    data_ids: List[UUID],
) -> int:
    """
    Track access to multiple Data entries efficiently.
    
    Uses batch processing with ON CONFLICT to handle multiple updates efficiently.
    
    Args:
        session: Database session
        data_ids: List of Data IDs to mark as accessed
    
    Returns:
        Number of successfully tracked entries
    
    Example:
        >>> count = await bulk_track_data_access(session, [id1, id2, id3])
    """
    if not data_ids:
        return 0
    
    success_count = 0
    
    try:
        # Process in batches to avoid overwhelming the database
        batch_size = 100
        for i in range(0, len(data_ids), batch_size):
            batch = data_ids[i:i + batch_size]
            
            # Prepare values for bulk insert
            values = []
            current_time = datetime.now(timezone.utc)
            
            for data_id in batch:
                values.append({
                    'data_id': data_id,
                    'last_accessed': current_time,
                    'access_count': 1,
                    'created_at': current_time,
                    'updated_at': current_time
                })
            
            # Use ON CONFLICT for efficient upsert
            stmt = pg_insert('data_access_tracking').values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=['data_id'],
                set_={
                    'last_accessed': current_time,
                    'access_count': 'data_access_tracking.access_count + 1',
                    'updated_at': current_time
                }
            )
            
            await session.execute(stmt)
            success_count += len(batch)
            
        return success_count
        
    except Exception as e:
        logger.error(f"Failed to bulk track Data access: {e}")
        return success_count


async def get_data_last_accessed(
    session: AsyncSession,
    data_id: UUID,
) -> Optional[datetime]:
    """
    Get the last accessed timestamp for a Data entry.
    
    Args:
        session: Database session
        data_id: ID of the Data entry
    
    Returns:
        Last accessed datetime or None if never accessed
    
    Example:
        >>> last_access = await get_data_last_accessed(session, data_id)
    """
    try:
        from sqlalchemy import text
        
        stmt = text(
            "SELECT last_accessed FROM data_access_tracking WHERE data_id = :data_id"
        )
        result = await session.execute(stmt, {'data_id': data_id})
        row = result.fetchone()
        
        return row[0] if row else None
        
    except Exception as e:
        logger.error(f"Failed to get last accessed time for Data {data_id}: {e}")
        return None


async def get_data_access_count(
    session: AsyncSession,
    data_id: UUID,
) -> int:
    """
    Get the access count for a Data entry.
    
    Args:
        session: Database session
        data_id: ID of the Data entry
    
    Returns:
        Number of times the Data has been accessed (0 if never accessed)
    
    Example:
        >>> count = await get_data_access_count(session, data_id)
    """
    try:
        from sqlalchemy import text
        
        stmt = text(
            "SELECT access_count FROM data_access_tracking WHERE data_id = :data_id"
        )
        result = await session.execute(stmt, {'data_id': data_id})
        row = result.fetchone()
        
        return row[0] if row else 0
        
    except Exception as e:
        logger.error(f"Failed to get access count for Data {data_id}: {e}")
        return 0
