"""Data-level cleanup operations.

This module provides functions to clean up unused Data entries based on access
tracking. By working at the Data level, it ensures proper cleanup of related
graph and vector database entries through existing deletion infrastructure.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_unused_data_ids(
    session: AsyncSession,
    days_threshold: int = 30,
) -> List[str]:
    """
    Get IDs of Data entries that haven't been accessed within the threshold.
    
    Args:
        session: Database session
        days_threshold: Number of days to consider data as unused
    
    Returns:
        List of Data UUIDs as strings that should be deleted
    
    Example:
        >>> unused_ids = await get_unused_data_ids(session, days_threshold=30)
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        
        # Query to find Data entries with old access timestamps
        # LEFT JOIN to include Data entries that have never been tracked
        from sqlalchemy import text
        
        query = text("""
            SELECT d.id::text
            FROM data d
            LEFT JOIN data_access_tracking dat ON d.id = dat.data_id
            WHERE 
                (dat.last_accessed IS NULL AND d.created_at < :cutoff_date)
                OR (dat.last_accessed < :cutoff_date)
        """)
        
        result = await session.execute(query, {'cutoff_date': cutoff_date})
        unused_ids = [row[0] for row in result.fetchall()]
        
        logger.info(f"Found {len(unused_ids)} unused Data entries (threshold: {days_threshold} days)")
        return unused_ids
        
    except Exception as e:
        logger.error(f"Failed to get unused Data IDs: {e}")
        return []


async def cleanup_unused_data(
    session: AsyncSession,
    days_threshold: int = 30,
    dry_run: bool = True,
) -> Dict[str, any]:
    """
    Clean up Data entries that haven't been accessed within the threshold.
    
    This function works at the Data level, allowing the existing deletion
    infrastructure to properly clean up related graph and vector database entries.
    
    Args:
        session: Database session
        days_threshold: Number of days to consider data as unused (default: 30)
        dry_run: If True, only report what would be deleted (default: True)
    
    Returns:
        Dictionary with cleanup results:
        {
            'success': bool,
            'dry_run': bool,
            'deleted_count': int,
            'unused_data_ids': List[str],
            'errors': List[str],
            'timestamp': datetime
        }
    
    Example:
        >>> # Preview what would be deleted
        >>> result = await cleanup_unused_data(session, days_threshold=30, dry_run=True)
        >>> print(f"Would delete {result['deleted_count']} Data entries")
        >>>
        >>> # Actually perform cleanup
        >>> result = await cleanup_unused_data(session, days_threshold=30, dry_run=False)
    """
    result = {
        'success': False,
        'dry_run': dry_run,
        'deleted_count': 0,
        'unused_data_ids': [],
        'errors': [],
        'timestamp': datetime.now(timezone.utc)
    }
    
    try:
        # Validate threshold
        if days_threshold < 0:
            raise ValueError("days_threshold must be non-negative")
        
        # Get unused Data IDs
        unused_ids = await get_unused_data_ids(session, days_threshold)
        result['unused_data_ids'] = unused_ids
        result['deleted_count'] = len(unused_ids)
        
        if not unused_ids:
            logger.info("No unused Data entries found")
            result['success'] = True
            return result
        
        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(unused_ids)} Data entries")
            result['success'] = True
            return result
        
        # Actually delete the Data entries
        # Note: The existing delete_data function should handle cascading
        # deletions to graph and vector databases
        from cognee.modules.data.deletion import delete_data_by_id
        
        deleted_count = 0
        for data_id in unused_ids:
            try:
                await delete_data_by_id(data_id, session)
                deleted_count += 1
            except Exception as e:
                error_msg = f"Failed to delete Data {data_id}: {e}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        await session.commit()
        
        result['deleted_count'] = deleted_count
        result['success'] = True
        logger.info(f"Successfully cleaned up {deleted_count} unused Data entries")
        
        return result
        
    except Exception as e:
        error_msg = f"Cleanup operation failed: {e}"
        logger.error(error_msg)
        result['errors'].append(error_msg)
        
        # Rollback on error
        await session.rollback()
        
        return result


async def get_cleanup_statistics(
    session: AsyncSession,
    days_threshold: int = 30,
) -> Dict[str, any]:
    """
    Get statistics about Data entries and their access patterns.
    
    Useful for determining appropriate cleanup thresholds and monitoring
    data usage patterns.
    
    Args:
        session: Database session
        days_threshold: Threshold for considering data as unused
    
    Returns:
        Dictionary with statistics:
        {
            'total_data_count': int,
            'tracked_count': int,
            'untracked_count': int,
            'unused_count': int,
            'active_count': int
        }
    
    Example:
        >>> stats = await get_cleanup_statistics(session, days_threshold=30)
        >>> print(f"Total: {stats['total_data_count']}, Unused: {stats['unused_count']}")
    """
    try:
        from sqlalchemy import text
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        
        # Get total Data count
        total_query = text("SELECT COUNT(*) FROM data")
        total_result = await session.execute(total_query)
        total_count = total_result.scalar()
        
        # Get tracked count
        tracked_query = text("SELECT COUNT(*) FROM data_access_tracking")
        tracked_result = await session.execute(tracked_query)
        tracked_count = tracked_result.scalar()
        
        # Get unused count
        unused_ids = await get_unused_data_ids(session, days_threshold)
        unused_count = len(unused_ids)
        
        return {
            'total_data_count': total_count,
            'tracked_count': tracked_count,
            'untracked_count': total_count - tracked_count,
            'unused_count': unused_count,
            'active_count': total_count - unused_count
        }
        
    except Exception as e:
        logger.error(f"Failed to get cleanup statistics: {e}")
        return {
            'total_data_count': 0,
            'tracked_count': 0,
            'untracked_count': 0,
            'unused_count': 0,
            'active_count': 0
        }
