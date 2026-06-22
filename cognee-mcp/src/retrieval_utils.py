"""Graph-backed document retrieval helpers for the Cognee MCP server."""

from __future__ import annotations

import json
from typing import Any, Literal


DOCUMENT_TYPES = {
    "Document",
    "TextDocument",
    "PdfDocument",
    "AudioDocument",
    "ImageDocument",
    "UnstructuredDocument",
    "CsvDocument",
    "DltRowDocument",
}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    return value


def normalize_node(node: Any) -> dict[str, Any]:
    """Normalize graph adapter node shapes into a plain dictionary."""
    node = _model_dump(node)
    if node is None:
        return {}

    if isinstance(node, dict):
        normalized = dict(node)
    else:
        try:
            normalized = dict(node.items()) if hasattr(node, "items") else dict(node)
        except (TypeError, ValueError):
            normalized = {
                key: getattr(node, key)
                for key in dir(node)
                if not key.startswith("_") and not callable(getattr(node, key))
            }

    properties = normalized.get("properties")
    if isinstance(properties, str) and properties:
        try:
            properties = json.loads(properties)
        except json.JSONDecodeError:
            properties = None

    if isinstance(properties, dict):
        normalized.pop("properties", None)
        normalized.update(properties)

    return normalized


def _node_id(node: Any) -> str | None:
    normalized = normalize_node(node)
    node_id = normalized.get("id")
    return str(node_id) if node_id is not None else None


def _node_type(node: Any) -> str:
    value = normalize_node(node).get("type")
    return str(value) if value is not None else ""


def _is_chunk_node(node: Any) -> bool:
    normalized = normalize_node(node)
    return _node_type(normalized) == "DocumentChunk" or (
        "chunk_index" in normalized and "text" in normalized
    )


def _is_document_node(node: Any) -> bool:
    normalized = normalize_node(node)
    node_type = _node_type(normalized)
    return (
        node_type in DOCUMENT_TYPES
        or node_type.endswith("Document")
        or ("raw_data_location" in normalized and not _is_chunk_node(normalized))
    )


def _relationship_name(edge: Any) -> str:
    edge = _model_dump(edge)
    if isinstance(edge, dict):
        value = edge.get("relationship_name") or edge.get("type") or edge.get("label")
        if value is None and isinstance(edge.get("properties"), dict):
            value = edge["properties"].get("relationship_name")
    else:
        value = edge
    return str(value or "")


def _is_relationship(edge: Any, expected: str) -> bool:
    rel = _relationship_name(edge).lower()
    expected = expected.lower()
    return rel == expected or f"relationship_name: {expected}" in rel


def _other_node(connection: Any, node_id: str) -> dict[str, Any] | None:
    if not isinstance(connection, (list, tuple)) or len(connection) < 3:
        return None

    first = normalize_node(connection[0])
    third = normalize_node(connection[2])
    if _node_id(first) == node_id:
        return third
    if _node_id(third) == node_id:
        return first
    return None


def _chunk_index(node: Any) -> int | None:
    value = normalize_node(node).get("chunk_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_text(node: Any) -> str:
    value = normalize_node(node).get("text")
    return str(value) if value is not None else ""


def _chunk_word_count(node: Any) -> int:
    normalized = normalize_node(node)
    for field_name in ("word_count", "chunk_size"):
        value = normalized.get(field_name)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return len(_chunk_text(node).split())


def _coerce_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


def _chunk_payload(node: Any, *, target_chunk_id: str | None = None) -> dict[str, Any]:
    normalized = normalize_node(node)
    chunk_id = _node_id(normalized)
    payload = {
        "chunk_id": chunk_id,
        "chunk_index": _chunk_index(normalized),
        "text": _chunk_text(normalized),
        "word_count": _chunk_word_count(normalized),
    }
    if target_chunk_id is not None:
        payload["is_target"] = chunk_id == target_chunk_id
    return payload


def _sort_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.get("chunk_index") is None,
            chunk.get("chunk_index") if chunk.get("chunk_index") is not None else 0,
            chunk.get("chunk_id") or "",
        ),
    )


def _subgraph_value(subgraph: Any, key: str, index: int) -> Any:
    subgraph = _model_dump(subgraph)
    if isinstance(subgraph, dict):
        return subgraph.get(key)
    if isinstance(subgraph, (list, tuple)) and len(subgraph) > index:
        return subgraph[index]
    return None


def _first_item(value: Any) -> Any:
    value = _model_dump(value)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _normalize_node_list(value: Any) -> list[dict[str, Any]]:
    value = _model_dump(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [normalize_node(item) for item in value if normalize_node(item)]


async def _find_parent_document(graph: Any, chunk_id: str) -> dict[str, Any] | None:
    connections = await graph.get_connections(chunk_id)
    for connection in connections:
        if not isinstance(connection, (list, tuple)) or len(connection) < 3:
            continue
        if not _is_relationship(connection[1], "is_part_of"):
            continue
        other = _other_node(connection, chunk_id)
        if other and _is_document_node(other):
            return other
    return None


async def _chunks_from_document_connections(graph: Any, document_id: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    connections = await graph.get_connections(document_id)
    for connection in connections:
        if not isinstance(connection, (list, tuple)) or len(connection) < 3:
            continue
        if not _is_relationship(connection[1], "is_part_of"):
            continue
        other = _other_node(connection, document_id)
        if other and _is_chunk_node(other):
            chunks.append(other)
    return chunks


async def _document_from_subgraph(
    graph: Any, document_id: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    get_document_subgraph = getattr(graph, "get_document_subgraph", None)
    if not callable(get_document_subgraph):
        return None, []

    try:
        subgraph = await get_document_subgraph(document_id)
    except Exception:
        return None, []

    if not subgraph:
        return None, []

    document = normalize_node(_first_item(_subgraph_value(subgraph, "document", 0)))
    chunks = _normalize_node_list(_subgraph_value(subgraph, "chunks", 1))
    return document or None, chunks


async def get_document_from_graph(
    graph: Any,
    document_id: str,
    *,
    include_metadata: bool = True,
    max_chunks: int = 0,
) -> dict[str, Any]:
    """Retrieve a document and its chunks through the graph adapter surface."""
    document_id = str(document_id).strip()
    max_chunks = _coerce_int(max_chunks, "max_chunks")
    if not document_id:
        raise ValueError("document_id must be a non-empty string.")
    if max_chunks < 0:
        raise ValueError("max_chunks must be greater than or equal to 0.")

    document, chunk_nodes = await _document_from_subgraph(graph, document_id)

    if not document:
        node = normalize_node(await graph.get_node(document_id))
        if _is_chunk_node(node):
            document = await _find_parent_document(graph, document_id)
        elif _is_document_node(node):
            document = node

    if not document:
        raise LookupError(f"Document not found: {document_id}")

    resolved_document_id = _node_id(document)
    if not resolved_document_id:
        raise LookupError(f"Document not found: {document_id}")

    if not chunk_nodes:
        chunk_nodes = await _chunks_from_document_connections(graph, resolved_document_id)

    all_chunks = _sort_chunks([_chunk_payload(chunk) for chunk in chunk_nodes])
    total_chunks = len(all_chunks)
    returned_chunks = all_chunks[:max_chunks] if max_chunks else all_chunks

    response: dict[str, Any] = {
        "document_id": resolved_document_id,
        "name": document.get("name") or "Unknown",
        "type": document.get("type") or "unknown",
        "chunk_count": len(returned_chunks),
        "total_chunks": total_chunks,
        "is_truncated": bool(max_chunks and total_chunks > max_chunks),
        "chunks": returned_chunks,
    }

    if include_metadata:
        excluded = {"id", "name", "type", "text", "chunk_index", "chunk_size", "word_count"}
        response["metadata"] = {
            key: value for key, value in document.items() if key not in excluded
        }

    return response


async def get_chunk_neighbors_from_graph(
    graph: Any,
    chunk_id: str,
    *,
    neighbor_count: int = 2,
    include_target: bool = True,
    direction: Literal["both", "forward", "backward"] = "both",
) -> dict[str, Any]:
    """Retrieve neighboring chunks from the same parent document."""
    chunk_id = str(chunk_id).strip()
    neighbor_count = _coerce_int(neighbor_count, "neighbor_count")
    include_target = _coerce_bool(include_target, "include_target")
    direction = str(direction).strip().lower()
    if not chunk_id:
        raise ValueError("chunk_id must be a non-empty string.")
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be at least 1.")
    if neighbor_count > 10:
        raise ValueError("neighbor_count must be less than or equal to 10.")
    if direction not in {"both", "forward", "backward"}:
        raise ValueError("direction must be one of: both, forward, backward.")

    target = normalize_node(await graph.get_node(chunk_id))
    if not target or not _is_chunk_node(target):
        raise LookupError(f"Chunk not found: {chunk_id}")

    target_index = _chunk_index(target)
    if target_index is None:
        raise LookupError(f"Chunk has no chunk_index: {chunk_id}")

    document = await _find_parent_document(graph, chunk_id)
    if not document:
        raise LookupError(f"Parent document not found for chunk: {chunk_id}")

    document_id = _node_id(document)
    if not document_id:
        raise LookupError(f"Parent document not found for chunk: {chunk_id}")

    document_result = await get_document_from_graph(
        graph,
        document_id,
        include_metadata=False,
        max_chunks=0,
    )

    if direction == "forward":
        min_index = target_index
        max_index = target_index + neighbor_count
    elif direction == "backward":
        min_index = target_index - neighbor_count
        max_index = target_index
    else:
        min_index = target_index - neighbor_count
        max_index = target_index + neighbor_count

    chunks = []
    for chunk in document_result["chunks"]:
        index = chunk.get("chunk_index")
        if index is None or index < min_index or index > max_index:
            continue
        is_target = chunk.get("chunk_id") == chunk_id
        if is_target and not include_target:
            continue
        chunks.append({**chunk, "is_target": is_target})

    return {
        "document_id": document_id,
        "document_name": document.get("name") or document_result.get("name") or "Unknown",
        "target_chunk_id": chunk_id,
        "target_chunk_index": target_index,
        "neighbor_count": neighbor_count,
        "direction": direction,
        "include_target": include_target,
        "chunks_returned": len(chunks),
        "chunks": _sort_chunks(chunks),
    }
