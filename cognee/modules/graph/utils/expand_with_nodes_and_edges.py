from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType, Timestamp
from cognee.modules.engine.utils import generate_edge_name, generate_node_name
from cognee.modules.engine.utils.timestamp_from_text import timestamp_from_text
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.shared.logging_utils import get_logger

logger = get_logger("expand_with_nodes_and_edges")

# What construction yields, keyed by str(id): the extracted entities, their
# types, and the timestamps facts anchor to.
GraphDataPoint = Entity | EntityType | Timestamp


def _strip_nonblank_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


def _get_or_create_timestamp(
    extracted_node: Node,
    data_chunk: DocumentChunk,
    data_points_by_id: dict[str, GraphDataPoint],
) -> Timestamp | None:
    """The ``Timestamp`` for a node the LLM typed "Timestamp", or None to keep it an Entity.

    A name that does not parse as a normalized time string stays an ordinary
    entity: it would be a Timestamp with no time. Edges are not a reason to
    demote — ``_with_timestamps_as_leaves`` has already turned every edge
    drawn from a parseable timestamp around. The id derives from the
    normalized string, so a time mentioned in several chunks is one node.
    """
    if generate_node_name(extracted_node.type) != "timestamp":
        return None
    timestamp = timestamp_from_text(extracted_node.name)
    if timestamp is None:
        logger.warning(
            "Timestamp %r is not a normalized time string; kept as an entity",
            extracted_node.name,
        )
        return None

    existing_data_point = data_points_by_id.get(str(timestamp.id))
    if isinstance(existing_data_point, Timestamp):
        return existing_data_point

    timestamp.importance_weight = data_chunk.importance_weight
    data_points_by_id[str(timestamp.id)] = timestamp
    return timestamp


def _get_or_create_entity_type(
    extracted_type: str,
    data_chunk: DocumentChunk,
    data_points_by_id: dict[str, GraphDataPoint],
) -> EntityType | None:
    """The chunk's EntityType for ``extracted_type``, or None when it has no name.

    An empty or punctuation-only type from the LLM would otherwise become an
    EntityType named "", which every reader shows as a type with no label.
    """
    normalized_type_name = generate_node_name(extracted_type)
    if not normalized_type_name.strip():
        return None
    entity_type_id = EntityType.id_for(extracted_type)
    entity_type_key = str(entity_type_id)
    existing_data_point = data_points_by_id.get(entity_type_key)
    if isinstance(existing_data_point, EntityType):
        return existing_data_point

    entity_type = EntityType(
        id=entity_type_id,
        name=normalized_type_name,
        description=normalized_type_name,
        importance_weight=data_chunk.importance_weight,
    )
    data_points_by_id[entity_type_key] = entity_type
    return entity_type


def _get_or_create_entity(
    extracted_node: Node,
    entity_id: UUID,
    entity_type: EntityType | None,
    data_chunk: DocumentChunk,
    data_points_by_id: dict[str, GraphDataPoint],
) -> Entity:
    entity_key = str(entity_id)
    existing_data_point = data_points_by_id.get(entity_key)
    if isinstance(existing_data_point, Entity):
        return existing_data_point

    entity = Entity(
        id=entity_id,
        name=generate_node_name(extracted_node.name),
        is_a=entity_type,
        description=extracted_node.description,
        belongs_to_set=data_chunk.belongs_to_set,
        importance_weight=data_chunk.importance_weight,
    )
    data_points_by_id[entity_key] = entity
    return entity


def _calculate_entity_ids_by_extracted_node_id(
    extracted_graph: KnowledgeGraph,
    data_chunk: DocumentChunk,
) -> dict[str, UUID]:
    """Calculate the final entity ID for every graph-local node ID.

    Multiple nodes with the same name remain distinct by receiving deterministic graph-scoped
    IDs instead of sharing one name-based ID.
    """
    extracted_node_ids: set[str] = set()
    nodes_by_name_based_id: dict[UUID, list[Node]] = {}
    for node in extracted_graph.nodes:
        if node.id in extracted_node_ids:
            raise ValueError(f"Duplicate node id in extracted graph: {node.id}")
        extracted_node_ids.add(node.id)

        name_based_entity_id = Entity.id_for(node.name)
        nodes_by_name_based_id.setdefault(name_based_entity_id, []).append(node)

    entity_ids_by_extracted_node_id: dict[str, UUID] = {}
    for name_based_entity_id, same_name_nodes in nodes_by_name_based_id.items():
        if len(same_name_nodes) == 1:
            entity_ids_by_extracted_node_id[same_name_nodes[0].id] = name_based_entity_id
            continue

        ordered_nodes = sorted(
            same_name_nodes,
            key=lambda node: (
                generate_node_name(node.type),
                generate_node_name(node.description),
                generate_node_name(node.id),
            ),
        )
        for ordinal, node in enumerate(ordered_nodes, start=1):
            entity_ids_by_extracted_node_id[node.id] = Entity.id_for(
                node.name,
                data_chunk.id,
                ordinal,
            )

    return entity_ids_by_extracted_node_id


def _link_chunk_to_entity(
    data_chunk: DocumentChunk,
    extracted_node: Node,
    entity: Entity | Timestamp,
) -> None:
    if data_chunk.contains is None:
        data_chunk.contains = []

    entity_description = _strip_nonblank_text(extracted_node.description)
    edge_text = (
        f"Document chunk mentions {entity.name}: {entity_description}"
        if entity_description
        else None
    )
    data_chunk.contains.append(
        (
            Edge(relationship_type="contains", edge_text=edge_text),
            entity,
        )
    )


def _with_timestamps_as_leaves(extracted_graph: KnowledgeGraph) -> KnowledgeGraph:
    """The extracted graph with every parseable timestamp on the target side of its edges.

    The prompt asks for timestamps as leaves (``marie_curie -born_at-> 1867``).
    When the LLM draws the edge the other way (``1898 -discovery_of-> radium``)
    the fact is still that radium is tied to 1898, so the edge is reversed and
    keeps its name instead of costing the chunk its timestamp: a Timestamp has
    no ``relations`` list to own an edge. An edge between two timestamps has
    no entity to live on and is dropped.
    """
    timestamp_ids = {
        node.id
        for node in extracted_graph.nodes
        if generate_node_name(node.type) == "timestamp"
        and timestamp_from_text(node.name) is not None
    }
    if not timestamp_ids:
        return extracted_graph

    edges = []
    for edge in extracted_graph.edges:
        if edge.source_node_id not in timestamp_ids:
            edges.append(edge)
        elif edge.target_node_id in timestamp_ids:
            logger.warning(
                "Dropping edge %r between timestamps %r and %r",
                edge.relationship_name,
                edge.source_node_id,
                edge.target_node_id,
            )
        else:
            logger.debug(
                "Reversing edge %r drawn from timestamp %r to %r",
                edge.relationship_name,
                edge.source_node_id,
                edge.target_node_id,
            )
            edges.append(
                edge.model_copy(
                    update={
                        "source_node_id": edge.target_node_id,
                        "target_node_id": edge.source_node_id,
                    }
                )
            )
    return extracted_graph.model_copy(update={"edges": edges})


def _convert_extracted_nodes_to_data_points(
    data_chunk: DocumentChunk,
    extracted_graph: KnowledgeGraph,
    data_points_by_id: dict[str, GraphDataPoint],
) -> dict[str, Entity | Timestamp]:
    """Construct final DataPoints and index entities by their graph-local LLM IDs."""
    entity_ids_by_extracted_node_id = _calculate_entity_ids_by_extracted_node_id(
        extracted_graph,
        data_chunk,
    )
    entities_by_extracted_node_id: dict[str, Entity | Timestamp] = {}

    for extracted_node in extracted_graph.nodes:
        timestamp = _get_or_create_timestamp(extracted_node, data_chunk, data_points_by_id)
        if timestamp is not None:
            entities_by_extracted_node_id[extracted_node.id] = timestamp
            _link_chunk_to_entity(data_chunk, extracted_node, timestamp)
            continue

        entity_type = _get_or_create_entity_type(
            extracted_node.type,
            data_chunk,
            data_points_by_id,
        )

        entity = _get_or_create_entity(
            extracted_node,
            entity_ids_by_extracted_node_id[extracted_node.id],
            entity_type,
            data_chunk,
            data_points_by_id,
        )
        entities_by_extracted_node_id[extracted_node.id] = entity
        _link_chunk_to_entity(data_chunk, extracted_node, entity)

    return entities_by_extracted_node_id


def _add_extracted_edges(
    data_chunk: DocumentChunk,
    extracted_graph: KnowledgeGraph,
    entities_by_extracted_node_id: dict[str, Entity | Timestamp],
    edges_by_identity: dict[EdgeIdentity, Edge],
) -> None:
    produced = data_chunk._produced_edge_identities
    for extracted_edge in extracted_graph.edges:
        source_entity = entities_by_extracted_node_id.get(extracted_edge.source_node_id)
        target_entity = entities_by_extracted_node_id.get(extracted_edge.target_node_id)
        if source_entity is None or target_entity is None:
            continue

        relationship_name = generate_edge_name(extracted_edge.relationship_name)
        edge_text = _strip_nonblank_text(extracted_edge.description)
        edge_identity = EdgeIdentity(
            source_id=str(source_entity.id),
            target_id=str(target_entity.id),
            relationship_name=relationship_name,
        )
        # Both records are written HERE, before the deduplication below, so a
        # relationship the graph already holds still counts for this chunk:
        # such an edge is never attached to the chunk's model, so neither
        # ownership nor evidence may depend on it being written.
        #
        # Ownership needs one entry per distinct relationship — it decides
        # what survives when a chunk is deleted.
        produced_key = (edge_identity.source_id, edge_identity.target_id, relationship_name)
        if produced_key not in produced:
            produced.append(produced_key)
        # Evidence needs every occurrence with its supporting text, so this
        # one is appended unconditionally.
        data_chunk._provenance_edges.append(
            (
                edge_identity.source_id,
                edge_identity.target_id,
                relationship_name,
                {"edge_text": edge_text},
            )
        )
        edges_by_identity.setdefault(
            edge_identity,
            Edge(
                relationship_type=relationship_name,
                edge_text=edge_text,
            ),
        )


def construct_data_points_and_edges(
    data_chunks: list[DocumentChunk],
    extracted_graphs: list[KnowledgeGraph],
) -> tuple[dict[str, GraphDataPoint], dict[EdgeIdentity, Edge]]:
    """Convert extracted knowledge graphs into DataPoints and edges.

    Nodes the LLM typed "Timestamp" become ``Timestamp`` datapoints (see
    ``_get_or_create_timestamp``), so the ``*_at`` edges pointing at them anchor
    facts to real times; every other node becomes an ``Entity`` with its
    ``EntityType``.
    """
    data_points_by_id: dict[str, GraphDataPoint] = {}
    edges_by_identity: dict[EdgeIdentity, Edge] = {}

    for data_chunk, extracted_graph in zip(data_chunks, extracted_graphs):
        if not extracted_graph:
            continue

        extracted_graph = _with_timestamps_as_leaves(extracted_graph)
        entities_by_extracted_node_id = _convert_extracted_nodes_to_data_points(
            data_chunk,
            extracted_graph,
            data_points_by_id,
        )
        _add_extracted_edges(
            data_chunk,
            extracted_graph,
            entities_by_extracted_node_id,
            edges_by_identity,
        )

    return data_points_by_id, edges_by_identity


def attach_new_edges_to_data_points(
    data_points_by_id: dict[str, GraphDataPoint],
    edges_by_identity: dict[EdgeIdentity, Edge],
    existing_edge_identities: set[EdgeIdentity],
) -> None:
    """Attach edges that are not already stored in the graph database."""
    for edge_identity, edge in edges_by_identity.items():
        if edge_identity in existing_edge_identities:
            continue

        source_data_point = data_points_by_id.get(edge_identity.source_id)
        target_data_point = data_points_by_id.get(edge_identity.target_id)
        if source_data_point is None or target_data_point is None:
            continue
        if isinstance(source_data_point, Timestamp):
            raise ValueError(
                f"Timestamp {source_data_point.timestamp_str!r} cannot own edge "
                f"{edge_identity.relationship_name!r}; construction must reverse it"
            )

        source_data_point.relations.append((edge, target_data_point))
