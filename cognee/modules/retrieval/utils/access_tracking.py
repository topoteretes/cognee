"""Utilities for tracking data access in retrievers."""

import json
from datetime import datetime, timezone
from typing import List, Any
from uuid import UUID
import os
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.shared.logging_utils import get_logger
from sqlalchemy import update
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph

logger = get_logger(__name__)


async def update_node_access_timestamps(items: List[Any]):
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":
        return

    if not items:
        return

    graph_engine = await get_graph_engine()
    timestamp_dt = datetime.now(timezone.utc)

    # Extract node IDs
    node_ids = []
    for item in items:
        item_id = item.payload.get("id") if hasattr(item, "payload") else item.get("id")
        if item_id:
            node_ids.append(str(item_id))

    if not node_ids:
        return

    # Focus on document-level tracking via projection
    try:
        doc_ids = await _find_origin_documents_via_projection(graph_engine, node_ids)
        if doc_ids:
            await _update_sql_records(doc_ids, timestamp_dt)
    except Exception as e:
        logger.error(f"Failed to update SQL timestamps: {e}")
        raise


async def _find_origin_documents_via_projection(graph_engine, node_ids):
    """Find origin documents using graph projection instead of DB queries"""
    # Project the entire graph with necessary properties
    memory_fragment = CogneeGraph()
    await memory_fragment.project_graph_from_db(
        graph_engine,
        node_properties_to_project=["id", "type"],
        edge_properties_to_project=["relationship_name"],
    )

    # Find origin documents by traversing the in-memory graph
    doc_ids = set()
    for node_id in node_ids:
        node = memory_fragment.get_node(node_id)
        if node and node.get_attribute("type") == "DocumentChunk":
            # Traverse edges to find connected documents
            for edge in node.get_skeleton_edges():
                # Get the neighbor node
                neighbor = (
                    edge.get_destination_node()
                    if edge.get_source_node().id == node_id
                    else edge.get_source_node()
                )
                if neighbor and neighbor.get_attribute("type") in ["TextDocument", "Document"]:
                    doc_ids.add(neighbor.id)

    return list(doc_ids)


async def _update_sql_records(doc_ids, timestamp_dt):
    """Update SQL Data table (same for all providers)"""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        stmt = (
            update(Data)
            .where(Data.id.in_([UUID(doc_id) for doc_id in doc_ids]))
            .values(last_accessed=timestamp_dt)
        )

        await session.execute(stmt)
        await session.commit()
