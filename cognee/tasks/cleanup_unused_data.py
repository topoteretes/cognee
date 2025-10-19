"""Periodic data cleanup functionality for Cognee.

This module implements automatic deletion of unused data in the memify pipeline
that hasn't been accessed for a specified period.

Issue: #1335 - Task to automatically delete data not accessed for specified time period
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.cleanup_unused_data import (
    delete_unused_data,
    get_unused_data_counts,
    get_table_statistics
)

logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    status: str
    deleted_counts: Dict[str, int]
    dry_run: bool
    timestamp: datetime
    errors: List[str]


async def cleanup_unused_data(
    days_threshold: int = 30,
    dry_run: bool = True,
    adapter=None,
    schema: Optional[str] = None
) -> CleanupResult:
    """Clean up data that hasn't been accessed within the threshold period.
    
    This function identifies and optionally deletes data from any graph database,
    including both default and custom graphs. It dynamically discovers tables
    with access tracking enabled.
    
    Args:
        days_threshold: Number of days without access to trigger deletion
        dry_run: If True, only report what would be deleted without deleting
        adapter: Database adapter to use (defaults to relational engine)
        schema: Optional schema name for custom graphs (e.g., 'my_custom_graph')
    
    Returns:
        CleanupResult containing deletion counts and status information
    
    Example:
        # Clean up default graph (dry run)
        result = await cleanup_unused_data(days_threshold=30, dry_run=True)
        
        # Clean up custom graph (actual deletion)
        result = await cleanup_unused_data(
            days_threshold=60,
            dry_run=False,
            schema='my_custom_graph'
        )
    """
    errors = []
    deleted_counts = {}
    
    try:
        logger.info(
            f"Starting {'DRY RUN' if dry_run else 'actual'} cleanup "
            f"for data older than {days_threshold} days"
            f"{f' in schema: {schema}' if schema else ''}"
        )
        
        # Get database adapter
        if adapter is None:
            adapter = get_relational_engine()
        
        async with adapter.get_async_session() as session:
            # Perform the cleanup using the schema-agnostic module
            deleted_counts = await delete_unused_data(
                session=session,
                days_threshold=days_threshold,
                schema=schema,
                dry_run=dry_run
            )
        
        total_deleted = sum(deleted_counts.values())
        status = "success"
        
        logger.info(
            f"Cleanup {'simulation' if dry_run else 'operation'} completed. "
            f"Total records {'would be deleted' if dry_run else 'deleted'}: {total_deleted}"
        )
        
    except Exception as e:
        logger.error(f"Error during cleanup operation: {e}", exc_info=True)
        errors.append(str(e))
        status = "error"
    
    return CleanupResult(
        status=status,
        deleted_counts=deleted_counts,
        dry_run=dry_run,
        timestamp=datetime.now(timezone.utc),
        errors=errors
    )


async def get_cleanup_preview(
    days_threshold: int = 30,
    adapter=None,
    schema: Optional[str] = None
) -> Dict[str, int]:
    """Preview how many records would be deleted without actually deleting.
    
    Args:
        days_threshold: Number of days without access to consider for deletion
        adapter: Database adapter to use (defaults to relational engine)
        schema: Optional schema name for custom graphs
    
    Returns:
        Dictionary mapping table names to count of records that would be deleted
    """
    try:
        if adapter is None:
            adapter = get_relational_engine()
        
        async with adapter.get_async_session() as session:
            counts = await get_unused_data_counts(
                session=session,
                days_threshold=days_threshold,
                schema=schema
            )
        
        return counts
    except Exception as e:
        logger.error(f"Error getting cleanup preview: {e}", exc_info=True)
        return {}


async def get_data_usage_statistics(
    adapter=None,
    schema: Optional[str] = None
) -> Dict[str, Dict]:
    """Get statistics about data usage across all tracked tables.
    
    Args:
        adapter: Database adapter to use (defaults to relational engine)
        schema: Optional schema name for custom graphs
    
    Returns:
        Dictionary with statistics for each tracked table
    """
    try:
        if adapter is None:
            adapter = get_relational_engine()
        
        async with adapter.get_async_session() as session:
            statistics = await get_table_statistics(
                session=session,
                schema=schema
            )
        
        return statistics
    except Exception as e:
        logger.error(f"Error getting data usage statistics: {e}", exc_info=True)
        return {}
