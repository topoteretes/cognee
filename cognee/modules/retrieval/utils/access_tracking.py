"""Utilities for tracking data access in retrievers."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID
import os
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.shared.logging_utils import get_logger
from sqlalchemy import update
from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
from cognee.modules.search.utils.transform_triplets_to_graph import transform_triplets_to_graph
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge

logger = get_logger(__name__)


async def update_node_access_timestamps(items: Any):
    if os.getenv("ENABLE_LAST_ACCESSED", "false").lower() != "true":
        return

    # In case there are no retrievable node ids, skip processing.
    if not items:
        return

    node_ids = _extract_access_node_ids(items)
    if not node_ids:
        logger.debug("No valid items to update access timestamps for.")
        return

    graph_engine = await get_graph_engine()
    timestamp_dt = datetime.now(timezone.utc)

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


def _extract_access_node_ids(items: Any) -> list[str]:
    if isinstance(items, list) and all(isinstance(item, Edge) for item in items):
        return _extract_node_ids_from_edges(items)
    if isinstance(items, dict):
        return _extract_node_ids_from_retrieval_dict(items)
    return []


def _extract_node_ids_from_edges(edges: list[Edge]) -> list[str]:
    graph_items = transform_triplets_to_graph(edges)
    node_ids = set()
    for item in graph_items.get("nodes"):
        item_id = item.payload.get("id") if hasattr(item, "payload") else item.get("id")
        item_id = _display_id(item_id)
        if item_id:
            node_ids.add(item_id)
    return sorted(node_ids)


def _extract_node_ids_from_retrieval_dict(items: dict) -> list[str]:
    node_ids = set()
    for chunk in items.get("chunks", []) or []:
        chunk_id = _result_id(chunk)
        if chunk_id:
            node_ids.add(chunk_id)

    for entity in items.get("entities", []) or []:
        if not isinstance(entity, dict):
            continue
        entity_id = _display_id(entity.get("id"))
        if entity_id:
            node_ids.add(entity_id)
        for edge in entity.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            for key in ("source_id", "target_id"):
                edge_node_id = _display_id(edge.get(key))
                if edge_node_id:
                    node_ids.add(edge_node_id)

    return sorted(node_ids)


def _result_id(result: Any) -> Optional[str]:
    payload = _payload(result)
    return _display_id(payload.get("id")) or _display_id(getattr(result, "id", None))


def _payload(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    payload = getattr(result, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _display_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
