"""Prepare edges for graph storage by ensuring each has default edge properties."""

from typing import Any, Dict, Iterable, List, Tuple

from cognee.modules.engine.utils import generate_edge_object_id


def _get_value(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)

    return getattr(item, field_name, None)


def _get_nonblank_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def get_edge_retrieval_text(edge_text: Any, relationship_name: Any) -> str:
    """Return the edge text used for retrieval, falling back to relationship_name."""
    return _get_nonblank_text(edge_text) or _get_nonblank_text(relationship_name) or ""


def _short_id(node_id: Any) -> str:
    return str(node_id)[:8]


def _trim_preview(text: str, max_length: int = 80) -> str:
    return " ".join(text.split())[:max_length]


def _get_node_label(node: Any, node_id: Any) -> str:
    short_id = _short_id(node_id)
    if node is None:
        return short_id

    name = _get_nonblank_text(_get_value(node, "name"))
    if name:
        return name

    type_name = type(node).__name__
    if type_name == "DocumentChunk":
        chunk_index = _get_value(node, "chunk_index")
        if chunk_index is not None:
            return f"chunk {chunk_index}"

        text = _get_nonblank_text(_get_value(node, "text"))
        if text:
            return _trim_preview(text)

    title = _get_nonblank_text(_get_value(node, "title"))
    if title:
        return title

    if type_name:
        return f"{type_name} {short_id}"

    return short_id


def _make_node_lookup(nodes: Iterable[Any] | None) -> dict[str, Any]:
    nodes_by_id = {}
    for node in nodes or []:
        node_id = _get_value(node, "id")
        if node_id is not None:
            nodes_by_id[str(node_id)] = node

    return nodes_by_id


def _build_fallback_edge_text(
    source_id: Any,
    target_id: Any,
    relationship_name: Any,
    nodes_by_id: dict[str, Any],
) -> str:
    source_label = _get_node_label(nodes_by_id.get(str(source_id)), source_id)
    target_label = _get_node_label(nodes_by_id.get(str(target_id)), target_id)
    relationship_phrase = get_edge_retrieval_text(None, relationship_name).replace("_", " ")

    if not relationship_phrase:
        relationship_phrase = "related to"

    return f"{source_label} {relationship_phrase} {target_label}."


def ensure_default_edge_properties(
    edges: List[Tuple[str, str, str, Dict[str, Any]]],
    nodes: Iterable[Any] | None = None,
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Ensure each edge has all default properties and retrieval text.
    Returns a new list of edges; does not mutate the input.
    """
    result = []
    nodes_by_id = _make_node_lookup(nodes)

    for source_id, target_id, relationship_name, properties in edges:
        props = dict(properties) if properties else {}
        if "edge_object_id" not in props:
            props["edge_object_id"] = generate_edge_object_id(
                source_id, target_id, relationship_name
            )
        if "feedback_weight" not in props:
            props["feedback_weight"] = 0.5

        edge_text = get_edge_retrieval_text(props.get("edge_text"), None)
        if not edge_text:
            props["edge_text"] = _build_fallback_edge_text(
                source_id,
                target_id,
                relationship_name,
                nodes_by_id,
            )

        result.append((source_id, target_id, relationship_name, props))
    return result
