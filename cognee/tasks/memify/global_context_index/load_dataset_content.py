"""Task 1 of the ``from_entities`` method: read what a dataset contains, in one graph call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData, Node
from cognee.infrastructure.engine import is_internal_node
from cognee.modules.data.methods import get_dataset_data
from cognee.modules.pipelines.models import PipelineContext

from .constants import DEFAULT_ENTITY_TYPE_NAME, DOCUMENT_INDEX_NODE_TYPES, ENTITY_NAME_MAX_CHARS
from .ids import dataset_id_from_context
from .models import DatasetContent, DatasetEntity


async def load_dataset_content(
    data: Any = None,
    ctx: PipelineContext | None = None,
) -> DatasetContent:
    """Memify extraction task: read everything the index needs in one graph call.

    The rows go straight into ``DatasetContent``. Building a ``CogneeGraph`` over
    them first would cost a pydantic model and a numpy array per node, an id hash
    per edge and an adjacency list we never traverse -- measured 58x slower and
    17x more memory on 14k nodes, and quadratic in the degree of an entity that
    appears in every chunk.
    """
    if data is not None and data != {} and data != [{}]:
        raise TypeError("load_dataset_content expected None, {}, or [{}].")

    dataset_id = dataset_id_from_context(ctx)
    if not dataset_id:
        # Bucket ids and the delete filter both key on it; without one a run would
        # write a parallel index that no later run can recognize.
        raise ValueError("The global context index requires a dataset context.")

    # Usually the graph holds this dataset alone. When datasets share one database
    # (backend access control off), its documents are the ones it has records for.
    document_ids = {str(data_item.id) for data_item in await get_dataset_data(UUID(dataset_id))}
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_filtered_graph_data(
        attribute_filters=[{"type": DOCUMENT_INDEX_NODE_TYPES}]
    )
    return _dataset_content_from_rows(nodes, edges, dataset_id, document_ids)


def _dataset_content_from_rows(
    nodes: list[Node],
    edges: list[EdgeData],
    dataset_id: str,
    document_ids: set[str],
) -> DatasetContent:
    """Reshape raw adapter rows -- ``(id, properties)`` and
    ``(source, target, relationship_name, properties)`` on every adapter."""
    graph_nodes = _parse_nodes(nodes, dataset_id, document_ids)
    graph_edges = _parse_edges(edges, graph_nodes)
    # Only entities a chunk mentions: entity ids derive from names, so in a
    # shared database one entity node serves every dataset that names it.
    mentioned = graph_edges.chunks_by_entity
    return DatasetContent(
        dataset_id=dataset_id,
        document_names=graph_nodes.document_names,
        document_by_chunk=graph_nodes.document_by_chunk,
        entities={
            entity_id: DatasetEntity(
                name=graph_nodes.entity_names[entity_id],
                type_name=graph_edges.type_by_entity.get(entity_id, DEFAULT_ENTITY_TYPE_NAME),
                chunk_ids=frozenset(chunk_ids),
            )
            for entity_id, chunk_ids in mentioned.items()
        },
        relations={
            (source_id, target_id): tuple(edge_texts)
            for (source_id, target_id), edge_texts in graph_edges.relations.items()
            if source_id in mentioned and target_id in mentioned
        },
        existing_summary_texts=graph_nodes.existing_summary_texts,
    )


@dataclass(frozen=True)
class _GraphNodes:
    """The node half of a dataset's content. ``entity_type_names`` only serves ``is_a`` edges."""

    document_names: dict[str, str]
    document_by_chunk: dict[str, str]
    entity_names: dict[str, str]
    entity_type_names: dict[str, str]
    existing_summary_texts: dict[str, str]


def _parse_nodes(nodes: list[Node], dataset_id: str, document_ids: set[str]) -> _GraphNodes:
    """The rows this index can name, by node type. A row it cannot use is skipped."""
    document_names: dict[str, str] = {}
    document_by_chunk: dict[str, str] = {}
    entity_names: dict[str, str] = {}
    entity_type_names: dict[str, str] = {}
    existing_summary_texts: dict[str, str] = {}

    for node_id, attributes in nodes:
        if is_internal_node(attributes):
            continue
        node_id = str(node_id)
        node_type = attributes.get("type")

        if node_type == "DocumentChunk":
            # A chunk with no document, or with another dataset's, counts nowhere.
            document_id = _normalized_text(attributes.get("document_id"))
            if document_id not in document_ids:
                continue
            document_by_chunk[node_id] = document_id
            document_name = _normalized_text(attributes.get("document_name"))
            if document_name:
                document_names.setdefault(document_id, document_name)
        elif node_type == "Entity":
            name = _normalized_text(attributes.get("name"), ENTITY_NAME_MAX_CHARS)
            if name:
                entity_names[node_id] = name
        elif node_type == "EntityType":
            name = _normalized_text(attributes.get("name"), ENTITY_NAME_MAX_CHARS)
            if name:
                entity_type_names[node_id] = name
        elif node_type == "GlobalContextSummary":
            # Only this dataset's index nodes, so a shared database never deletes
            # another dataset's. A node with no dataset_id did not come from here.
            node_dataset_id = attributes.get("dataset_id")
            if node_dataset_id is not None and str(node_dataset_id) == dataset_id:
                existing_summary_texts[node_id] = str(attributes.get("text") or "")

    return _GraphNodes(
        document_names=document_names,
        document_by_chunk=document_by_chunk,
        entity_names=entity_names,
        entity_type_names=entity_type_names,
        existing_summary_texts=existing_summary_texts,
    )


@dataclass(frozen=True)
class _GraphEdges:
    """The edge half: what the node rows alone cannot say."""

    chunks_by_entity: dict[str, set[str]]  # which chunks mention an entity
    type_by_entity: dict[str, str]  # what an entity is
    relations: dict[tuple[str, str], list[str]]  # what holds between two of them


def _parse_edges(edges: list[EdgeData], graph_nodes: _GraphNodes) -> _GraphEdges:
    """The edges between the nodes ``_parse_nodes`` kept; any other edge is skipped."""
    document_by_chunk = graph_nodes.document_by_chunk
    entity_names = graph_nodes.entity_names
    entity_type_names = graph_nodes.entity_type_names
    chunks_by_entity: dict[str, set[str]] = {}
    type_by_entity: dict[str, str] = {}
    relations: dict[tuple[str, str], list[str]] = {}

    for source_id, target_id, edge_relationship_name, attributes in edges:
        source_id = str(source_id)
        target_id = str(target_id)
        name = attributes.get("relationship_name") or edge_relationship_name

        if name == "contains" and source_id in document_by_chunk and target_id in entity_names:
            chunks_by_entity.setdefault(target_id, set()).add(source_id)
        elif name == "is_a" and source_id in entity_names and target_id in entity_type_names:
            # Several types: keep the one that sorts first, so the pick is stable.
            type_name = entity_type_names[target_id]
            current_type_name = type_by_entity.get(source_id)
            if current_type_name is None or type_name < current_type_name:
                type_by_entity[source_id] = type_name
        elif source_id in entity_names and target_id in entity_names:
            edge_text = _normalized_text(attributes.get("edge_text"))
            if edge_text:
                relations.setdefault((source_id, target_id), []).append(edge_text)

    return _GraphEdges(
        chunks_by_entity=chunks_by_entity,
        type_by_entity=type_by_entity,
        relations=relations,
    )


def _normalized_text(value: Any, max_chars: int | None = None) -> str | None:
    """Collapsed whitespace, optionally capped; ``None`` when nothing is left."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if max_chars is not None:
        text = text[:max_chars]
    return text or None
