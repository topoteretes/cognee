import re
import unicodedata
from typing import Any, Optional

from cognee.modules.retrieval.hybrid.facts import connection_edge_type_id
from cognee.modules.retrieval.hybrid.results import (
    display_value,
    first_display_value,
    payload,
    result_id,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


async def build_entities(
    graph_engine: Any,
    entity_hits: list[Any],
    max_edges_per_entity: int,
    edge_ranks: Optional[dict[str, int]] = None,
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
):
    if not entity_hits:
        return []

    if node_name:
        entity_hits = [
            result
            for result in entity_hits
            if _node_matches_filter(payload(result), node_name, node_name_filter_operator)
        ]

    entities = [_entity_from_result(result) for result in entity_hits]
    entity_ids = [entity["id"] for entity in entities if entity["id"]]
    if not entity_ids:
        return entities

    try:
        nodes, edges = await graph_engine.get_neighborhood(entity_ids, depth=1)
    except Exception as error:
        logger.warning(
            "Graph neighborhood retrieval failed; returning entities without edges: %s", error
        )
        return entities

    connections_by_entity_id = _partition_neighborhood(
        entity_ids,
        nodes,
        edges,
        node_name=node_name,
        node_name_filter_operator=node_name_filter_operator,
    )
    for entity in entities:
        entity["edges"] = _edge_bullets_from_connections(
            connections_by_entity_id.get(entity["id"], []),
            max_edges_per_entity,
            edge_ranks or {},
        )
    return entities


def _partition_neighborhood(
    entity_ids: list[str],
    nodes: list[Any],
    edges: list[Any],
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
) -> dict[str, list[tuple[dict, dict, dict]]]:
    """Rebuild per-entity (source, edge, target) connection triples from the flat one-hop
    subgraph returned by get_neighborhood; drops neighbor-to-neighbor edges."""
    nodes_by_id = {}
    for node in nodes or []:
        if isinstance(node, (list, tuple)) and len(node) == 2 and isinstance(node[1], dict):
            nodes_by_id[str(node[0])] = {"id": str(node[0]), **node[1]}

    connections = {entity_id: [] for entity_id in entity_ids}
    for edge in edges or []:
        if not isinstance(edge, (list, tuple)) or len(edge) < 3:
            continue
        source_id, target_id = str(edge[0]), str(edge[1])
        properties = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
        source = nodes_by_id.get(source_id, {"id": source_id})
        target = nodes_by_id.get(target_id, {"id": target_id})
        if node_name and not _connection_matches_node_filter(
            source,
            properties,
            target,
            entity_ids,
            node_name,
            node_name_filter_operator,
        ):
            continue
        triple = (
            source,
            {"relationship_name": edge[2], "properties": properties},
            target,
        )
        if source_id in connections:
            connections[source_id].append(triple)
        if target_id in connections and target_id != source_id:
            connections[target_id].append(triple)
    return connections


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


def _edge_bullets_from_connections(
    connections: list[Any], max_edges: int, edge_ranks: dict[str, int]
) -> list[dict]:
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
        normalized_text = _normalized_text(bullet["text"])
        if not dedupe_key and normalized_text in seen_texts:
            continue

        if dedupe_key:
            seen_keys.add(dedupe_key)
        else:
            seen_texts.add(normalized_text)
        edges.append(bullet)
    edges.sort(key=lambda edge: _edge_sort_key(edge, edge_ranks))
    return edges[:max_edges]


def _edge_sort_key(edge: dict, edge_ranks: dict[str, int]) -> tuple[int, int, int]:
    """Query-ranked evidence first; type edges only break ties for unranked edges."""
    rank = edge_ranks.get(edge.get("edge_type_id"))
    if rank is not None:
        return (0, rank, 0)
    return (1, 0 if _is_type_edge(edge) else 1, 0)


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

    properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
    return {
        "text": text,
        "source": source_label,
        "target": target_label,
        "source_id": display_value(source.get("id")),
        "relationship": relationship,
        "target_id": display_value(target.get("id")),
        "edge_type_id": connection_edge_type_id(edge),
        "provenance": _edge_provenance(properties),
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


def _connection_matches_node_filter(
    source: dict,
    properties: dict,
    target: dict,
    entity_ids: list[str],
    node_name: list[str],
    node_name_filter_operator: str,
) -> bool:
    seed_ids = set(entity_ids)
    source_matches = str(source.get("id")) in seed_ids or _node_matches_filter(
        source, node_name, node_name_filter_operator
    )
    target_matches = str(target.get("id")) in seed_ids or _node_matches_filter(
        target, node_name, node_name_filter_operator
    )
    if not source_matches or not target_matches:
        return False

    # Some graph providers attach scope directly to an edge. When present it
    # must agree with the requested scope; absent edge tags inherit the strict
    # endpoint decision above.
    if "belongs_to_set" in properties:
        return _node_matches_filter(properties, node_name, node_name_filter_operator)
    return True


def _node_matches_filter(node: dict, node_name: list[str], node_name_filter_operator: str) -> bool:
    values = node.get("belongs_to_set")
    if not isinstance(values, list):
        return False

    actual = {_node_set_name(value) for value in values}
    actual.discard(None)
    expected = {str(value) for value in node_name}
    if node_name_filter_operator.upper() == "AND":
        return expected.issubset(actual)
    return bool(actual & expected)


def _node_set_name(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        return first_display_value(value.get("name"), value.get("id"))
    return first_display_value(getattr(value, "name", None), value)


_PROVENANCE_KEYS = {
    "source_id",
    "source_chunk_id",
    "data_id",
    "dataset_id",
    "document_id",
    "source_path",
    "page",
    "page_number",
    "start",
    "end",
    "timestamp",
    "valid_from",
    "valid_to",
    "valid_at",
    "created_at",
    "updated_at",
}


def _edge_provenance(properties: dict) -> dict:
    return {
        key: value
        for key, value in properties.items()
        if key in _PROVENANCE_KEYS and value is not None
    }


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()
