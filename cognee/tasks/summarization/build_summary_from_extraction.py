from uuid import uuid5

from pydantic import BaseModel

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.graph.utils import ensure_default_edge_properties, get_graph_from_model
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.summarization.models import TextSummary


async def build_summary_from_extraction(
    chunk: DocumentChunk, graph_model: type[BaseModel]
) -> TextSummary | None:
    """Summarize a chunk by its extracted types and relations, without an LLM.

    One line per extracted type comes before the relations. Returns None when the
    chunk's extraction produced no relations.
    """
    nodes, edges = await _collect_extracted_graph(chunk, graph_model)
    edges = ensure_default_edge_properties(edges, nodes=nodes)
    relation_lines = dict.fromkeys(properties["edge_text"] for *_, properties in edges)
    if not relation_lines:
        return None
    type_lines = _format_type_lines(nodes)
    return TextSummary(
        id=uuid5(chunk.id, "TextSummary"),
        made_from=chunk,
        source_chunk_id=str(chunk.id),
        belongs_to_set=chunk.belongs_to_set,
        text="\n".join((*type_lines, *relation_lines)),
        importance_weight=chunk.importance_weight,
    )


async def _collect_extracted_graph(chunk: DocumentChunk, graph_model: type[BaseModel]):
    """The chunk's own extracted nodes and relations."""
    if issubclass(graph_model, KnowledgeGraph):
        # The chunk's extraction record. Its entities' relations would also hold
        # edges other chunks in the batch attached, so they are not walked.
        entities = [entity for _, entity in chunk.contains or []]
        return entities, chunk._provenance_edges
    # A custom model is stored by walking the chunk, so walk it the same way. The
    # chunk, its document and its node sets are stored around the extraction, not
    # extracted, so drop them and every edge that touches them.
    nodes, edges = await get_graph_from_model(chunk)
    node_set_ids = (getattr(node_set, "id", None) for node_set in chunk.belongs_to_set or [])
    skipped_ids = {chunk.id, chunk.is_part_of.id, *node_set_ids}
    nodes = [node for node in nodes if node.id not in skipped_ids]
    edges = [edge for edge in edges if edge[0] not in skipped_ids and edge[1] not in skipped_ids]
    return nodes, edges


def _format_type_lines(nodes) -> list[str]:
    """One ``Type: name, name`` line per extracted type, in first-seen order."""
    names_by_type: dict[str, dict[str, None]] = {}
    for node in nodes:
        label = _get_node_label(node)
        type_name = _get_type_name(node)
        if not label or not type_name:
            continue
        names_by_type.setdefault(type_name, {}).setdefault(label, None)
    return [f"{type_name}: {', '.join(names)}" for type_name, names in names_by_type.items()]


def _get_type_name(node) -> str | None:
    """The extracted type. An Entity stores it on is_a; a custom node stores it on type."""
    is_a = getattr(node, "is_a", None)
    entity_type = _strip_nonblank_text(getattr(is_a, "name", None))
    return entity_type or _strip_nonblank_text(getattr(node, "type", None))


def _get_node_label(node) -> str | None:
    """The label storage's fallback edge text uses: the first index field, then the name."""
    metadata = getattr(node, "metadata", None) or {}
    for field_name in [*(metadata.get("index_fields") or []), "name"]:
        value = _strip_nonblank_text(getattr(node, field_name, None))
        if value:
            return " ".join(value.split())[:80]
    return None


def _strip_nonblank_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
