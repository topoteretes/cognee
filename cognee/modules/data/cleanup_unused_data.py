"""Schema-agnostic data cleanup operations.

This module provides functions to clean up unused data from any graph database,
including both default and custom graphs. It dynamically discovers tables using
SQLAlchemy metadata and performs cleanup based on last_accessed timestamps.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from sqlalchemy import MetaData, Table, select, delete, inspect, Column
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)


def discover_tracked_tables(metadata: MetaData, schema: Optional[str] = None) -> List[Table]:
    """Dynamically discover all tables that have last_accessed column.
    
    Args:
        metadata: SQLAlchemy MetaData instance with reflected tables
        schema: Optional schema name for custom graphs
    
    Returns:
        List of Table objects that have last_accessed column
    """
    tracked_tables = []
    
    for table_name, table in metadata.tables.items():
        # Filter by schema if specified
        if schema and table.schema != schema:
            continue
        
        # Check if table has last_accessed column
        if 'last_accessed' in table.columns:
            tracked_tables.append(table)
            logger.debug(f"Found tracked table: {table_name}")
    
    return tracked_tables


async def get_unused_data_counts(
    session: AsyncSession,
    days_threshold: int = 30,
    schema: Optional[str] = None
) -> Dict[str, int]:
    """Get count of unused records per table.
    
    Args:
        session: Database session
        days_threshold: Number of days to consider data unused
        schema: Optional schema name for custom graphs
    
    Returns:
        Dictionary mapping table names to count of unused records
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
    counts = {}
    
    # Reflect database metadata
    metadata = MetaData(schema=schema)
    await session.run_sync(lambda sync_session: metadata.reflect(bind=sync_session.bind))
    
    # Discover tables with tracking
    tracked_tables = discover_tracked_tables(metadata, schema)
    
    for table in tracked_tables:
        try:
            # Build query to count unused records
            query = select(table.c.id).where(
                (table.c.last_accessed < cutoff_date) |
                (table.c.last_accessed == None)
            )
            result = await session.execute(query)
            count = len(result.all())
            counts[table.name] = count
            logger.info(f"Table {table.name}: {count} unused records")
        except Exception as e:
            logger.error(f"Error counting unused data in {table.name}: {e}")
            counts[table.name] = 0
    
    return counts


async def delete_unused_data(
    session: AsyncSession,
    days_threshold: int = 30,
    schema: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """Delete unused data from all tracked tables.
    
    Args:
        session: Database session
        days_threshold: Number of days to consider data unused
        schema: Optional schema name for custom graphs
        dry_run: If True, only count records without deleting
    
    Returns:
        Dictionary mapping table names to count of deleted records
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
    deleted_counts = {}
    
    # Reflect database metadata
    metadata = MetaData(schema=schema)
    await session.run_sync(lambda sync_session: metadata.reflect(bind=sync_session.bind))
    
    # Discover tables with tracking
    tracked_tables = discover_tracked_tables(metadata, schema)
    
    logger.info(f"{'DRY RUN: ' if dry_run else ''}Cleaning up data older than {days_threshold} days")
    
    for table in tracked_tables:
        try:
            # Build delete query
            delete_query = delete(table).where(
                (table.c.last_accessed < cutoff_date) |
                (table.c.last_accessed == None)
            )
            
            if dry_run:
                # Just count records that would be deleted
                count_query = select(table.c.id).where(
                    (table.c.last_accessed < cutoff_date) |
                    (table.c.last_accessed == None)
                )
                result = await session.execute(count_query)
                count = len(result.all())
            else:
                # Execute delete
                result = await session.execute(delete_query)
                count = result.rowcount
                await session.commit()
            
            deleted_counts[table.name] = count
            logger.info(
                f"{'Would delete' if dry_run else 'Deleted'} {count} records from {table.name}"
            )
        except Exception as e:
            logger.error(f"Error deleting unused data from {table.name}: {e}")
            deleted_counts[table.name] = 0
            if not dry_run:
                await session.rollback()
    
    return deleted_counts


async def get_table_statistics(
    session: AsyncSession,
    schema: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Get statistics for all tracked tables.
    
    Args:
        session: Database session
        schema: Optional schema name for custom graphs
    
    Returns:
        Dictionary mapping table names to their statistics
    """
    statistics = {}
    
    # Reflect database metadata
    metadata = MetaData(schema=schema)
    await session.run_sync(lambda sync_session: metadata.reflect(bind=sync_session.bind))
    
    # Discover tables with tracking
    tracked_tables = discover_tracked_tables(metadata, schema)
    
    for table in tracked_tables:
        try:
            # Total count
            total_query = select(table.c.id)
            total_result = await session.execute(total_query)
            total_count = len(total_result.all())
            
            # Never accessed count
            never_accessed_query = select(table.c.id).where(
                table.c.last_accessed == None
            )
            never_result = await session.execute(never_accessed_query)
            never_accessed = len(never_result.all())
            
            # Recently accessed (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_query = select(table.c.id).where(
                table.c.last_accessed >= recent_date
            )
            recent_result = await session.execute(recent_query)
            recently_accessed = len(recent_result.all())
            
            statistics[table.name] = {
                'total_records': total_count,
                'never_accessed': never_accessed,
                'recently_accessed': recently_accessed,
                'accessed_percentage': round(
                    ((total_count - never_accessed) / total_count * 100) if total_count > 0 else 0,
                    2
                )
            }
            logger.info(f"Statistics for {table.name}: {statistics[table.name]}")
        except Exception as e:
            logger.error(f"Error getting statistics for {table.name}: {e}")
            statistics[table.name] = {
                'total_records': 0,
                'never_accessed': 0,
                'recently_accessed': 0,
                'accessed_percentage': 0
            }
    
    return statistics
