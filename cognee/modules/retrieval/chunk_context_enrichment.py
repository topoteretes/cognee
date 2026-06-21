"""Graph-backed enrichment for chunk search results (parent document + neighbors)."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from cognee.infrastructure.databases.graph.utils import normalize_graph_result
from cognee.shared.logging_utils import get_logger

logger = get_logger("ChunkContextEnrichment")

MAX_EXPAND_NEIGHBORS = 10


def _safe_chunk_id(raw_id: Any) -> Optional[str]:
    """Validate chunk IDs before graph queries to avoid injection via malformed IDs."""
    if raw_id is None:
        return None
    try:
        return str(UUID(str(raw_id).strip()))
    except (ValueError, TypeError, AttributeError):
        return None


def extract_chunk_ids(found_chunks: list[Any]) -> tuple[list[str], dict[str, Any]]:
    chunk_ids: list[str] = []
    chunk_id_map: dict[str, Any] = {}

    for chunk in found_chunks:
        chunk_id = None
        if hasattr(chunk, "id"):
            chunk_id = _safe_chunk_id(chunk.id)
        elif hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
            chunk_id = _safe_chunk_id(chunk.payload.get("id"))
        elif isinstance(chunk, dict):
            if "id" in chunk:
                chunk_id = _safe_chunk_id(chunk["id"])
            elif isinstance(chunk.get("payload"), dict):
                chunk_id = _safe_chunk_id(chunk["payload"].get("id"))

        if chunk_id:
            chunk_ids.append(chunk_id)
            chunk_id_map[chunk_id] = chunk

    return chunk_ids, chunk_id_map


def _build_parent_info(row: dict[str, Any]) -> dict[str, str]:
    parent_info = {
        "id": str(row.get("doc_id", "")),
        "name": row.get("doc_name") or "Unknown",
    }
    if row.get("doc_type"):
        parent_info["type"] = str(row["doc_type"])
    return parent_info


async def fetch_parent_documents(graph_engine: Any, chunk_ids: list[str]) -> dict[str, dict]:
    parent_map: dict[str, dict] = {}
    if not chunk_ids:
        return parent_map

    try:
        query = """
        MATCH (chunk:DocumentChunk)-[:is_part_of]->(doc:Document)
        WHERE chunk.id IN $chunk_ids
        RETURN chunk.id as chunk_id, doc.id as doc_id, doc.name as doc_name, doc.type as doc_type
        """
        result = await graph_engine.query(query, params={"chunk_ids": chunk_ids})
        result = normalize_graph_result(result, ["chunk_id", "doc_id", "doc_name", "doc_type"])
        for row in result:
            chunk_id = _safe_chunk_id(row.get("chunk_id"))
            if chunk_id:
                parent_map[chunk_id] = _build_parent_info(row)
    except Exception as error:
        logger.warning("Batched parent-document lookup failed: %s", error)

    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in parent_map]
    for chunk_id in missing:
        try:
            query = """
            MATCH (chunk:DocumentChunk {id: $chunk_id})-[:is_part_of]->(doc:Document)
            RETURN doc.id as doc_id, doc.name as doc_name, doc.type as doc_type
            LIMIT 1
            """
            result = await graph_engine.query(query, params={"chunk_id": chunk_id})
            result = normalize_graph_result(result, ["doc_id", "doc_name", "doc_type"])
            if result:
                parent_map[chunk_id] = _build_parent_info(result[0])
        except Exception as error:
            logger.debug("Individual parent lookup failed for %s: %s", chunk_id, error)

    return parent_map


async def fetch_neighbor_chunks(
    graph_engine: Any,
    chunk_ids: list[str],
    *,
    neighbor_count: int,
) -> dict[str, list[dict[str, Any]]]:
    if neighbor_count <= 0 or not chunk_ids:
        return {}

    neighbor_count = min(max(neighbor_count, 1), MAX_EXPAND_NEIGHBORS)
    neighbors_by_chunk: dict[str, list[dict[str, Any]]] = {chunk_id: [] for chunk_id in chunk_ids}

    try:
        query = """
        MATCH (target:DocumentChunk)-[:is_part_of]->(doc:Document)
        WHERE target.id IN $chunk_ids
        MATCH (doc)<-[:is_part_of]-(sibling:DocumentChunk)
        RETURN target.id as chunk_id, target.chunk_index as target_index,
               sibling.id as sibling_id, sibling.chunk_index as sibling_index,
               sibling.text as sibling_text
        """
        result = await graph_engine.query(query, params={"chunk_ids": chunk_ids})
        result = normalize_graph_result(
            result,
            ["chunk_id", "target_index", "sibling_id", "sibling_index", "sibling_text"],
        )
    except Exception as error:
        logger.warning("Neighbor chunk lookup failed: %s", error)
        return neighbors_by_chunk

    for row in result:
        chunk_id = _safe_chunk_id(row.get("chunk_id"))
        sibling_id = _safe_chunk_id(row.get("sibling_id"))
        target_index = row.get("target_index")
        sibling_index = row.get("sibling_index")
        if chunk_id is None or sibling_id is None or target_index is None or sibling_index is None:
            continue
        if chunk_id not in neighbors_by_chunk:
            continue
        if abs(int(sibling_index) - int(target_index)) > neighbor_count:
            continue
        if sibling_id == chunk_id:
            continue
        neighbors_by_chunk[chunk_id].append(
            {
                "chunk_id": sibling_id,
                "chunk_index": int(sibling_index),
                "text": row.get("sibling_text") or "",
            }
        )

    for chunk_id, neighbors in neighbors_by_chunk.items():
        neighbors.sort(key=lambda item: item.get("chunk_index", 0))

    return neighbors_by_chunk


def _attach_to_payload(chunk: Any, key: str, value: Any) -> None:
    if hasattr(chunk, "payload") and isinstance(chunk.payload, dict):
        chunk.payload[key] = value
    elif isinstance(chunk, dict):
        if isinstance(chunk.get("payload"), dict):
            chunk["payload"][key] = value
        else:
            chunk[key] = value


async def enrich_chunk_results(
    graph_engine: Any,
    found_chunks: list[Any],
    *,
    expand_neighbors: int = 0,
    strict_enrichment: bool = False,
) -> list[Any]:
    if not found_chunks:
        return found_chunks

    chunk_ids, chunk_id_map = extract_chunk_ids(found_chunks)
    if not chunk_ids:
        if strict_enrichment:
            raise ValueError("No valid chunk IDs found for enrichment")
        logger.warning("No valid chunk IDs found, skipping enrichment")
        return found_chunks

    parent_map = await fetch_parent_documents(graph_engine, chunk_ids)
    neighbor_map = await fetch_neighbor_chunks(
        graph_engine, chunk_ids, neighbor_count=expand_neighbors
    )

    enriched_count = 0
    for chunk_id, chunk in chunk_id_map.items():
        parent_info = parent_map.get(chunk_id)
        if parent_info:
            _attach_to_payload(chunk, "parent_document", parent_info)
            _attach_to_payload(
                chunk,
                "is_part_of",
                {
                    "id": parent_info["id"],
                    "name": parent_info.get("name"),
                    "type": parent_info.get("type"),
                },
            )
            enriched_count += 1
        elif strict_enrichment:
            raise ValueError(f"No parent document found for chunk {chunk_id}")

        if expand_neighbors > 0:
            neighbors = neighbor_map.get(chunk_id, [])
            if neighbors:
                _attach_to_payload(chunk, "neighboring_chunks", neighbors)

    if found_chunks:
        success_rate = enriched_count / len(found_chunks) * 100
        logger.info(
            "Enriched %s/%s chunks with parent document metadata (%.1f%%)",
            enriched_count,
            len(found_chunks),
            success_rate,
        )

    return found_chunks
