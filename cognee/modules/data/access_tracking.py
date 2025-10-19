"""Schema-agnostic access tracking utilities.

This module provides functions to track when data entities are accessed,
enabling efficient cleanup of unused data across all graph databases,
including both default and custom graphs.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Union, Set
from uuid import UUID
from sqlalchemy import MetaData, Table, update, inspect
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def get_tracked_tables(metadata: MetaData, schema: Optional[str] = None) -> Set[str]:
    """Get names of all tables that have last_accessed column.
    
    Args:
        metadata: SQLAlchemy MetaData instance with reflected tables
        schema: Optional schema name for custom graphs
    
    Returns:
        Set of table names that support access tracking
    """
    tracked_tables = set()
    
    for table_name, table in metadata.tables.items():
        # Filter by schema if specified
        if schema and table.schema != schema:
            continue
        
        # Check if table has last_accessed column
        if 'last_accessed' in table.columns:
            tracked_tables.add(table.name)
    
    return tracked_tables


async def update_last_accessed(
    session: AsyncSession,
    entity_ids: Union[UUID, List[UUID]],
    table_name: str,
    schema: Optional[str] = None
) -> None:
    """Update the last_accessed timestamp for specified entities.
    
    Args:
        session: Database session
        entity_ids: Single entity ID or list of entity IDs to update
        table_name: Name of the table to update
        schema: Optional schema name for custom graphs
    
    Returns:
        None
    
    Raises:
        ValueError: If table doesn't exist or doesn't have last_accessed column
    """
    # Normalize to list
    if not isinstance(entity_ids, list):
        entity_ids = [entity_ids]
    
    if not entity_ids:
        return
    
    try:
        # Reflect table metadata
        metadata = MetaData(schema=schema)
        await session.run_sync(lambda sync_session: metadata.reflect(bind=sync_session.bind))
        
        # Get the table
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        if full_table_name not in metadata.tables:
            raise ValueError(f"Table {full_table_name} not found")
        
        table = metadata.tables[full_table_name]
        
        # Verify table has last_accessed column
        if 'last_accessed' not in table.columns:
            raise ValueError(f"Table {full_table_name} does not have last_accessed column")
        
        # Build and execute update query
        stmt = (
            update(table)
            .where(table.c.id.in_(entity_ids))
            .values(last_accessed=datetime.now(timezone.utc))
        )
        
        result = await session.execute(stmt)
        await session.commit()
        
        logger.debug(
            f"Updated last_accessed for {result.rowcount} records in {full_table_name}"
        )
    except Exception as e:
        logger.error(f"Error updating last_accessed for {table_name}: {e}")
        await session.rollback()
        raise


async def bulk_update_last_accessed(
    session: AsyncSession,
    updates: List[dict],
    schema: Optional[str] = None
) -> None:
    """Perform bulk update of last_accessed timestamps across multiple tables.
    
    Args:
        session: Database session
        updates: List of dicts with 'table_name' and 'entity_ids' keys
        schema: Optional schema name for custom graphs
    
    Example:
        updates = [
            {'table_name': 'document_chunks', 'entity_ids': [id1, id2]},
            {'table_name': 'entities', 'entity_ids': [id3, id4]}
        ]
    """
    for update_info in updates:
        table_name = update_info.get('table_name')
        entity_ids = update_info.get('entity_ids', [])
        
        if table_name and entity_ids:
            try:
                await update_last_accessed(
                    session=session,
                    entity_ids=entity_ids,
                    table_name=table_name,
                    schema=schema
                )
            except Exception as e:
                logger.error(f"Error in bulk update for {table_name}: {e}")
                # Continue with other updates even if one fails
                continue


async def mark_entity_accessed(
    session: AsyncSession,
    entity_id: UUID,
    table_name: str,
    schema: Optional[str] = None
) -> bool:
    """Mark a single entity as accessed.
    
    Convenience function for updating a single entity.
    
    Args:
        session: Database session
        entity_id: Entity ID to mark as accessed
        table_name: Name of the table
        schema: Optional schema name for custom graphs
    
    Returns:
        True if successful, False otherwise
    """
    try:
        await update_last_accessed(
            session=session,
            entity_ids=[entity_id],
            table_name=table_name,
            schema=schema
        )
        return True
    except Exception as e:
        logger.error(f"Failed to mark entity {entity_id} as accessed: {e}")
        return False
