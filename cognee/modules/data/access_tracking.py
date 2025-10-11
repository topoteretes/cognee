"""Access tracking utilities for updating last_accessed timestamps on data entities.

This module provides hooks to track when document chunks, entities, summaries, 
associations, and metadata are accessed, enabling efficient cleanup of unused data.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Union
from uuid import UUID

logger = logging.getLogger(__name__)


async def update_last_accessed(
    entity_ids: Union[UUID, List[UUID]],
    entity_type: str,
    adapter = None
) -> None:
    """Update the last_accessed timestamp for specified entities.
    
    Args:
        entity_ids: Single entity ID or list of entity IDs to update
        entity_type: Type of entity ('chunk', 'entity', 'summary', 'association', 'metadata')
        adapter: Database adapter instance. If None, will get from infrastructure.
    
    Returns:
        None
    
    Raises:
        ValueError: If entity_type is not recognized
    """
    if adapter is None:
        from cognee.infrastructure.databases.relational import get_relational_engine
        adapter = get_relational_engine()
    
    # Normalize to list
    if not isinstance(entity_ids, list):
        entity_ids = [entity_ids]
    
    if not entity_ids:
        return
    
    # Map entity types to table names
    table_mapping = {
        'chunk': 'DocumentChunk',
        'entity': 'Entity',
        'summary': 'DocumentSummary',
        'association': 'Association',
        'metadata': 'Metadata'
    }
    
    if entity_type not in table_mapping:
        raise ValueError(f"Unknown entity type: {entity_type}")
    
    table_name = table_mapping[entity_type]
    current_time = datetime.now(timezone.utc)
    
    try:
        # Update timestamps in batch for efficiency
        query = f"""
            UPDATE {table_name}
            SET last_accessed = :timestamp
            WHERE id IN :ids
        """
        
        await adapter.execute(
            query,
            {"timestamp": current_time, "ids": tuple(entity_ids)}
        )
        
        logger.debug(
            f"Updated last_accessed for {len(entity_ids)} {entity_type} entities"
        )
    except Exception as e:
        logger.warning(
            f"Failed to update last_accessed for {entity_type}: {str(e)}"
        )


async def track_chunk_access(chunk_ids: Union[UUID, List[UUID]], adapter = None) -> None:
    """Track access to document chunks."""
    await update_last_accessed(chunk_ids, 'chunk', adapter)


async def track_entity_access(entity_ids: Union[UUID, List[UUID]], adapter = None) -> None:
    """Track access to entities."""
    await update_last_accessed(entity_ids, 'entity', adapter)


async def track_summary_access(summary_ids: Union[UUID, List[UUID]], adapter = None) -> None:
    """Track access to summaries."""
    await update_last_accessed(summary_ids, 'summary', adapter)


async def track_association_access(association_ids: Union[UUID, List[UUID]], adapter = None) -> None:
    """Track access to associations."""
    await update_last_accessed(association_ids, 'association', adapter)


async def track_metadata_access(metadata_ids: Union[UUID, List[UUID]], adapter = None) -> None:
    """Track access to metadata."""
    await update_last_accessed(metadata_ids, 'metadata', adapter)
