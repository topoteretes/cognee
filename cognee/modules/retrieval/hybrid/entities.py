import asyncio
from typing import Any, Optional

from cognee.modules.retrieval.hybrid.results import (
    display_value,
    first_display_value,
    payload,
    result_id,
)


async def build_entities(graph_engine: Any, entity_hits: list[Any], max_edges_per_entity: int):
    if not entity_hits:
        return []

    entities = [_entity_from_result(result) for result in entity_hits]
    if await graph_engine.is_empty():
        return entities

    connections_by_entity = await asyncio.gather(
        *[graph_engine.get_connections(entity["id"]) for entity in entities]
    )
    for entity, connections in zip(entities, connections_by_entity):
        entity["edges"] = _edge_bullets_from_connections(connections, max_edges_per_entity)
    return entities


def format_entities(entities: list[dict]) -> str:
    blocks = []
    for entity in entities or []:
        block = _format_entity(entity)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "## Relevant entities\n" + "\n\n".join(blocks)


def _entity_from_result(result: Any) -> dict:
    result_payload = payload(result)
    entity_id = result_id(result) or ""
    return {
        "id": entity_id,
        "name": first_display_value(
            result_payload.get("name"), result_payload.get("text"), entity_id
        )
        or "",
        "description": display_value(result_payload.get("description")),
        "type": _entity_type(result_payload),
        "edges": [],
    }


def _format_entity(entity: dict) -> str:
    name = display_value(entity.get("name"))
    if not name:
        return ""

    entity_type = _entity_type(entity)
    header = f"### {name} ({entity_type})" if entity_type else f"### {name}"

    lines = [header]
    description = display_value(entity.get("description"))
    if description:
        lines.append(description)

    for edge in entity.get("edges", []):
        edge_text = display_value(edge.get("text"))
        if edge_text:
            lines.append(f"- {edge_text}")

    return "\n".join(lines)


def _entity_type(result_payload: dict) -> Optional[str]:
    for value in (result_payload.get("is_a"), result_payload.get("type")):
        entity_type = display_value(value)
        if entity_type and entity_type not in {"IndexSchema"}:
            return entity_type
    return None


def _edge_bullets_from_connections(connections: list[Any], max_edges: int) -> list[dict]:
    if max_edges <= 0:
        return []

    edges = []
    seen_keys = set()
    seen_texts = set()
    for connection in connections or []:
        unpacked = _unpack_connection(connection)
        if unpacked is None:
            continue

        source, edge, target = unpacked
        bullet = _edge_bullet(source, edge, target)
        if not bullet:
            continue

        dedupe_key = _edge_dedupe_key(bullet)
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if not dedupe_key and bullet["text"] in seen_texts:
            continue

        if dedupe_key:
            seen_keys.add(dedupe_key)
        else:
            seen_texts.add(bullet["text"])
        edges.append(bullet)
    edges.sort(key=lambda edge: 0 if _is_type_edge(edge) else 1)
    return edges[:max_edges]


def _unpack_connection(connection: Any) -> Optional[tuple[dict, dict, dict]]:
    if not isinstance(connection, (list, tuple)) or len(connection) != 3:
        return None
    source, edge, target = connection
    if not isinstance(source, dict) or not isinstance(edge, dict) or not isinstance(target, dict):
        return None
    return source, edge, target


def _edge_bullet(source: dict, edge: dict, target: dict) -> Optional[dict]:
    source_label = _node_label(source)
    target_label = _node_label(target)
    relationship = display_value(edge.get("relationship_name"))
    text = first_display_value(edge.get("edge_text"), _nested_edge_text(edge))
    if not text and source_label and relationship and target_label:
        text = f"{source_label} -- {relationship} -- {target_label}"
    if not text:
        return None

    return {
        "text": text,
        "source": source_label,
        "target": target_label,
        "source_id": display_value(source.get("id")),
        "relationship": relationship,
        "target_id": display_value(target.get("id")),
    }


def _edge_dedupe_key(edge: dict) -> Optional[tuple[str, str, str]]:
    source_id = display_value(edge.get("source_id"))
    relationship = display_value(edge.get("relationship"))
    target_id = display_value(edge.get("target_id"))
    if source_id and relationship and target_id:
        return source_id, relationship, target_id
    return None


def _is_type_edge(edge: dict) -> bool:
    relationship = display_value(edge.get("relationship"))
    if relationship:
        normalized = relationship.lower().replace("_", " ").replace("-", " ").strip()
        if normalized == "is a":
            return True

    text = display_value(edge.get("text"))
    return bool(text and " is a " in f" {text.lower()} ")


def _nested_edge_text(edge: dict) -> Optional[str]:
    properties = edge.get("properties")
    if not isinstance(properties, dict):
        return None
    return display_value(properties.get("edge_text"))


def _node_label(node: dict) -> Optional[str]:
    return first_display_value(node.get("name"), node.get("id"))
