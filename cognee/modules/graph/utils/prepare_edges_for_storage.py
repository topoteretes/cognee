"""Prepare edges for graph storage by ensuring each has default edge properties."""

from typing import Any, Dict, Iterable, List, Tuple

from cognee.modules.engine.utils import generate_edge_object_id
from cognee.shared.logging_utils import get_logger

logger = get_logger()


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


def _trim_preview(text: str, max_length: int = 80) -> str:
    return " ".join(text.split())[:max_length]


def _get_node_label(node: Any, node_id: Any) -> str:
    """Return a human-readable label for a node, based on the author's
    declared `metadata["index_fields"]`. Falls back to `name` if declared,
    then to the class name as a last resort.

    Structural DataPoints (e.g. `Timestamp`) intentionally declare empty
    `index_fields` and have no `name`. For those we emit the class name
    rather than raising, since they are part of the graph but were never
    intended to carry an embeddable identifier.
    """
    if node is None:
        logger.warning(
            "Cannot resolve node %r to build a fallback edge_text label; "
            "falling back to the node id. Pass the endpoint nodes alongside "
            "the edges to ensure_default_edge_properties.",
            node_id,
        )
        return str(node_id)

    metadata = _get_value(node, "metadata") or {}
    for field in metadata.get("index_fields") or []:
        value = _get_nonblank_text(_get_value(node, field))
        if value:
            return _trim_preview(value)

    name = _get_nonblank_text(_get_value(node, "name"))
    if name:
        return name

    type_name = type(node).__name__
    logger.warning(
        "Falling back to class name %r as the edge_text label for node %r: "
        "neither `metadata['index_fields']` nor `name` yields a value.",
        type_name,
        node_id,
    )
    return type_name


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


# Node provenance fields copied onto an edge from its source endpoint so edges
# are traceable to the same source lineage as their nodes (issue #3632).
_EDGE_PROVENANCE_FIELDS = (
    "source_pipeline",
    "source_task",
    "source_node_set",
    "source_content_hash",
    "source_user",
    "source_dataset_id",
    "source_document_id",
    "source_chunk_id",
)


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

        # Provenance-by-default (issue #3632): nodes are stamped with their
        # source lineage during the pipeline run, but edges were left with
        # empty provenance. Copy the source endpoint's lineage onto the edge
        # (set-if-absent, so explicit edge values win) so every edge is
        # traceable to the same source as its nodes. When provenance is
        # disabled the source node carries no lineage, so nothing is copied.
        source_node = nodes_by_id.get(str(source_id))
        if source_node is not None:
            for field_name in _EDGE_PROVENANCE_FIELDS:
                if field_name in props:
                    continue
                value = _get_value(source_node, field_name)
                if value is not None:
                    props[field_name] = value

        result.append((source_id, target_id, relationship_name, props))
    return result
