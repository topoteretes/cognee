"""Periodic data cleanup functionality for Cognee.

This module implements automatic deletion of unused Data entries that haven't
been accessed for a specified period. By working at the Data level, it ensures
proper cleanup of related graph and vector database entries.

Issue: #1335 - Task to automatically delete data not accessed for specified time period
"""
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.cleanup_unused_data import (
    cleanup_unused_data as cleanup_data_module,
    get_cleanup_statistics
)

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    success: bool
    deleted_count: int
    dry_run: bool
    timestamp: datetime
    errors: list[str]


async def cleanup_unused_data(
    days_threshold: int = 30,
    dry_run: bool = True,
    adapter=None
) -> CleanupResult:
    """
    Clean up Data entries that haven't been accessed within the threshold.
    
    This is the main task function that wraps the Data-level cleanup module.
    It works at the Data model level, ensuring proper cascade deletion of
    related graph and vector database entries through existing deletion infrastructure.
    
    Args:
        days_threshold: Number of days without access to consider for deletion (default: 30)
        dry_run: If True, only preview deletions without making changes (default: True)
        adapter: Database adapter to use (defaults to relational engine)
    
    Returns:
        CleanupResult with operation details
    
    Raises:
        ValueError: If days_threshold is negative
    
    Examples:
        >>> # Preview what would be deleted
        >>> result = await cleanup_unused_data(days_threshold=30, dry_run=True)
        >>> print(f"Would delete {result.deleted_count} Data entries")
        >>>
        >>> # Actually perform cleanup
        >>> result = await cleanup_unused_data(days_threshold=30, dry_run=False)
        >>> if result.success:
        >>>     print(f"Successfully deleted {result.deleted_count} Data entries")
    """
    try:
        if adapter is None:
            adapter = get_relational_engine()
        
        async with adapter.get_async_session() as session:
            # Call the Data-level cleanup module
            result_dict = await cleanup_data_module(
                session=session,
                days_threshold=days_threshold,
                dry_run=dry_run
            )
            
            # Convert module result to CleanupResult dataclass
            return CleanupResult(
                success=result_dict['success'],
                deleted_count=result_dict['deleted_count'],
                dry_run=result_dict['dry_run'],
                timestamp=result_dict['timestamp'],
                errors=result_dict['errors']
            )
    
    except Exception as e:
        logger.error(f"Cleanup operation failed: {e}", exc_info=True)
        return CleanupResult(
            success=False,
            deleted_count=0,
            dry_run=dry_run,
            timestamp=datetime.now(timezone.utc),
            errors=[str(e)]
        )


async def get_data_usage_statistics(
    days_threshold: int = 30,
    adapter=None
) -> Dict[str, Any]:
    """
    Get statistics about Data entries and their access patterns.
    
    Useful for monitoring data usage and determining appropriate cleanup thresholds.
    
    Args:
        days_threshold: Threshold for considering data as unused (default: 30)
        adapter: Database adapter to use (defaults to relational engine)
    
    Returns:
        Dictionary with statistics:
        {
            'total_data_count': int,        # Total number of Data entries
            'tracked_count': int,           # Number of Data entries with access tracking
            'untracked_count': int,         # Number of Data entries never accessed
            'unused_count': int,            # Number of Data entries exceeding threshold
            'active_count': int             # Number of Data entries within threshold
        }
    
    Example:
        >>> stats = await get_data_usage_statistics(days_threshold=30)
        >>> print(f"Total Data: {stats['total_data_count']}")
        >>> print(f"Unused Data: {stats['unused_count']}")
        >>> print(f"Active Data: {stats['active_count']}")
    """
    try:
        if adapter is None:
            adapter = get_relational_engine()
        
        async with adapter.get_async_session() as session:
            statistics = await get_cleanup_statistics(
                session=session,
                days_threshold=days_threshold
            )
        
        return statistics
    
    except Exception as e:
        logger.error(f"Error getting data usage statistics: {e}", exc_info=True)
        return {
            'total_data_count': 0,
            'tracked_count': 0,
            'untracked_count': 0,
            'unused_count': 0,
            'active_count': 0
        }
