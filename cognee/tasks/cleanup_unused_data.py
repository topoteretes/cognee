"""Periodic data cleanup functionality for Cognee.

This module implements automatic deletion of unused data in the memify pipeline
that hasn't been accessed for a specified period.

Issue: #1335 - Task to automatically delete data not accessed for specified time period
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

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
    adapter = None
) -> CleanupResult:
    """Clean up data that hasn't been accessed within the threshold period.
    
    This function identifies and optionally deletes:
    - Document Chunks
    - Entities
    - Summaries
    - Associations
    - Metadata
    
    that haven't been accessed for more than `days_threshold` days.
    
    Args:
        days_threshold: Number of days after which unused data is considered for deletion.
                       Default is 30 days.
        dry_run: If True, only reports what would be deleted without actually deleting.
                Default is True for safety.
        adapter: Database adapter instance. If None, will get from infrastructure.
    
    Returns:
        CleanupResult: Object containing status, counts of deleted items, and any errors.
    
    Example:
        >>> result = await cleanup_unused_data(days_threshold=30, dry_run=True)
        >>> print(f"Would delete {result.deleted_counts['document_chunks']} chunks")
    """
    logger.info(
        f"Starting cleanup with threshold={days_threshold} days, dry_run={dry_run}"
    )
    
    deleted_counts = {
        "document_chunks": 0,
        "entities": 0,
        "summaries": 0,
        "associations": 0,
        "metadata": 0,
    }
    errors = []
    
    # Get database adapter
    if adapter is None:
        try:
            from cognee.infrastructure.databases.relational import get_relational_engine
            adapter = get_relational_engine()
        except Exception as e:
            logger.error(f"Failed to get database adapter: {str(e)}")
            errors.append(f"Database connection error: {str(e)}")
            return CleanupResult(
                status="error",
                deleted_counts=deleted_counts,
                dry_run=dry_run,
                timestamp=datetime.now(timezone.utc),
                errors=errors,
            )
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
    logger.info(f"Cutoff date for cleanup: {cutoff_date}")
    
    try:
        # Query and delete document chunks
        try:
            count_query = """
                SELECT COUNT(*) FROM DocumentChunk
                WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
            """
            result = await adapter.execute(count_query, {"cutoff_date": cutoff_date})
            chunk_count = result.scalar() if hasattr(result, 'scalar') else 0
            
            logger.info(f"Found {chunk_count} document chunks for cleanup")
            
            if not dry_run and chunk_count > 0:
                delete_query = """
                    DELETE FROM DocumentChunk
                    WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
                """
                await adapter.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_counts["document_chunks"] = chunk_count
                logger.info(f"Deleted {chunk_count} document chunks")
            else:
                deleted_counts["document_chunks"] = chunk_count
        except Exception as e:
            logger.warning(f"Error cleaning document chunks: {str(e)}")
            errors.append(f"document_chunks: {str(e)}")
        
        # Query and delete entities
        try:
            count_query = """
                SELECT COUNT(*) FROM Entity
                WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
            """
            result = await adapter.execute(count_query, {"cutoff_date": cutoff_date})
            entity_count = result.scalar() if hasattr(result, 'scalar') else 0
            
            logger.info(f"Found {entity_count} entities for cleanup")
            
            if not dry_run and entity_count > 0:
                delete_query = """
                    DELETE FROM Entity
                    WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
                """
                await adapter.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_counts["entities"] = entity_count
                logger.info(f"Deleted {entity_count} entities")
            else:
                deleted_counts["entities"] = entity_count
        except Exception as e:
            logger.warning(f"Error cleaning entities: {str(e)}")
            errors.append(f"entities: {str(e)}")
        
        # Query and delete summaries
        try:
            count_query = """
                SELECT COUNT(*) FROM DocumentSummary
                WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
            """
            result = await adapter.execute(count_query, {"cutoff_date": cutoff_date})
            summary_count = result.scalar() if hasattr(result, 'scalar') else 0
            
            logger.info(f"Found {summary_count} summaries for cleanup")
            
            if not dry_run and summary_count > 0:
                delete_query = """
                    DELETE FROM DocumentSummary
                    WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
                """
                await adapter.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_counts["summaries"] = summary_count
                logger.info(f"Deleted {summary_count} summaries")
            else:
                deleted_counts["summaries"] = summary_count
        except Exception as e:
            logger.warning(f"Error cleaning summaries: {str(e)}")
            errors.append(f"summaries: {str(e)}")
        
        # Query and delete associations
        try:
            count_query = """
                SELECT COUNT(*) FROM Association
                WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
            """
            result = await adapter.execute(count_query, {"cutoff_date": cutoff_date})
            assoc_count = result.scalar() if hasattr(result, 'scalar') else 0
            
            logger.info(f"Found {assoc_count} associations for cleanup")
            
            if not dry_run and assoc_count > 0:
                delete_query = """
                    DELETE FROM Association
                    WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
                """
                await adapter.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_counts["associations"] = assoc_count
                logger.info(f"Deleted {assoc_count} associations")
            else:
                deleted_counts["associations"] = assoc_count
        except Exception as e:
            logger.warning(f"Error cleaning associations: {str(e)}")
            errors.append(f"associations: {str(e)}")
        
        # Query and delete metadata
        try:
            count_query = """
                SELECT COUNT(*) FROM Metadata
                WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
            """
            result = await adapter.execute(count_query, {"cutoff_date": cutoff_date})
            meta_count = result.scalar() if hasattr(result, 'scalar') else 0
            
            logger.info(f"Found {meta_count} metadata items for cleanup")
            
            if not dry_run and meta_count > 0:
                delete_query = """
                    DELETE FROM Metadata
                    WHERE last_accessed < :cutoff_date OR last_accessed IS NULL
                """
                await adapter.execute(delete_query, {"cutoff_date": cutoff_date})
                deleted_counts["metadata"] = meta_count
                logger.info(f"Deleted {meta_count} metadata items")
            else:
                deleted_counts["metadata"] = meta_count
        except Exception as e:
            logger.warning(f"Error cleaning metadata: {str(e)}")
            errors.append(f"metadata: {str(e)}")
        
        if dry_run:
            logger.info("Dry run mode - no data was actually deleted")
        
        status = "success" if not errors else "partial_success"
        
    except Exception as e:
        logger.error(f"Unexpected error during cleanup: {str(e)}")
        errors.append(f"Unexpected error: {str(e)}")
        status = "error"
    
    result = CleanupResult(
        status=status,
        deleted_counts=deleted_counts,
        dry_run=dry_run,
        timestamp=datetime.now(timezone.utc),
        errors=errors,
    )
    
    total_deleted = sum(deleted_counts.values())
    logger.info(
        f"Cleanup completed: status={status}, "
        f"{'would delete' if dry_run else 'deleted'}={total_deleted} items"
    )
    
    return result


# Design Notes:
# - All database tables should have a 'last_accessed' timestamp field
# - Access tracking is implemented via cognee.modules.data.access_tracking module
# - Hooks are integrated in retrieval functions to update timestamps
# - This function can be scheduled to run periodically (e.g., via cron or task scheduler)
# - Default behavior is dry_run=True for safety
# - Consider adding configuration options for:
#   - Enabling/disabling automatic cleanup
#   - Customizing thresholds per data type
#   - Notification/alerting for cleanup operations
