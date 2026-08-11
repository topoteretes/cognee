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
    """Find origin documents using targeted neighborhood queries instead of DB graph projections"""
    doc_ids = set()

    if not node_ids:
        return []

    # Get immediate neighborhood instead of pulling the entire graph into memory
    nodes, edges = await graph_engine.get_neighborhood(list(node_ids), depth=1)

    # Handle node tuple variations across different graph adapters
    node_type_map = {}
    for n in nodes:
        if isinstance(n, (list, tuple)) and len(n) >= 2 and isinstance(n[1], dict):
            node_type_map[str(n[0])] = n[1].get("type")

    for edge in edges:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue

        # Extract source and target IDs defensively (handles Ladybug vs Neo4j vs Kuzu formats)
        src = edge[0]
        tgt = edge[2] if isinstance(edge[1], str) and isinstance(edge[2], dict) else edge[1] # Handle Ladybug (dict, str, dict)
        
        source_id = str(src.get("id", src) if isinstance(src, dict) else src)
        target_id = str(tgt.get("id", tgt) if isinstance(tgt, dict) else tgt)

        # Check both directions for chunk-to-document links
        if source_id in node_ids and node_type_map.get(source_id) == "DocumentChunk":
            if node_type_map.get(target_id) in ["TextDocument", "Document"]:
                doc_ids.add(target_id)
        elif target_id in node_ids and node_type_map.get(target_id) == "DocumentChunk":
            if node_type_map.get(source_id) in ["TextDocument", "Document"]:
                doc_ids.add(source_id)

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
