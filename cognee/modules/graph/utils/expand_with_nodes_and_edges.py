from typing import Optional

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_name,
)
from cognee.modules.ontology.graph_enrichment import (
    canonicalize_graphs,
    extend_graph_with_ontology,
)
from cognee.shared.data_models import KnowledgeGraph


def _create_node_key(node_id: str, category: str) -> str:
    """Create a standardized node key."""
    return f"{node_id}_{category}"


def _create_edge_key(source_id: str, target_id: str, relationship_name: str) -> str:
    """Create a standardized edge key."""
    return f"{source_id}_{target_id}_{relationship_name}"


def _strip_nonblank_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


def _create_type_node(
    node_type: str,
    added_nodes_map: dict,
    data_chunk: DocumentChunk,
) -> EntityType:
    """Create or retrieve a type node."""
    node_id = EntityType.id_for(node_type)
    node_name = generate_node_name(node_type)
    type_node_key = _create_node_key(node_id, "type")

    if type_node_key in added_nodes_map:
        return added_nodes_map[type_node_key]

    type_node = EntityType(
        id=node_id,
        name=node_name,
        description=node_name,
        importance_weight=data_chunk.importance_weight,
    )
    added_nodes_map[type_node_key] = type_node
    return type_node


def _create_entity_node(
    node_id: str,
    node_name: str,
    node_description: str,
    type_node: EntityType,
    added_nodes_map: dict,
    data_chunk: DocumentChunk,
) -> Entity:
    """Create or retrieve an entity node."""
    generated_node_id = Entity.id_for(node_id)
    generated_node_name = generate_node_name(node_name)
    entity_node_key = _create_node_key(generated_node_id, "entity")

    if entity_node_key in added_nodes_map:
        return added_nodes_map[entity_node_key]

    entity_node = Entity(
        id=generated_node_id,
        name=generated_node_name,
        is_a=type_node,
        description=node_description,
        belongs_to_set=data_chunk.belongs_to_set,
        importance_weight=data_chunk.importance_weight,
    )
    added_nodes_map[entity_node_key] = entity_node
    return entity_node


def _process_graph_nodes(
    data_chunk: DocumentChunk,
    graph: KnowledgeGraph,
    added_nodes_map: dict,
) -> None:
    """Process nodes in a knowledge graph."""
    for node in graph.nodes:
        type_node = _create_type_node(node.type, added_nodes_map, data_chunk)
        entity_node = _create_entity_node(
            node.id,
            node.name,
            node.description,
            type_node,
            added_nodes_map,
            data_chunk,
        )

        if data_chunk.contains is None:
            data_chunk.contains = []

        entity_description = _strip_nonblank_text(node.description)
        edge_text = (
            f"Document chunk mentions {entity_node.name}: {entity_description}"
            if entity_description
            else None
        )

        data_chunk.contains.append(
            (
                Edge(
                    relationship_type="contains",
                    edge_text=edge_text,
                ),
                entity_node,
            )
        )


def _process_graph_edges(
    graph: KnowledgeGraph, existing_edges_map: dict, relationships: list
) -> None:
    """Process edges in a knowledge graph."""
    for edge in graph.edges:
        source_node_id = Entity.id_for(edge.source_node_id)
        target_node_id = Entity.id_for(edge.target_node_id)
        relationship_name = generate_edge_name(edge.relationship_name)
        edge_key = _create_edge_key(source_node_id, target_node_id, relationship_name)
        edge_text = _strip_nonblank_text(edge.description)

        if edge_key not in existing_edges_map:
            relationships.append(
                (
                    source_node_id,
                    target_node_id,
                    relationship_name,
                    {
                        "relationship_name": relationship_name,
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "ontology_valid": False,
                        "edge_text": edge_text,
                    },
                )
            )
            existing_edges_map[edge_key] = True


def _resolve_node(node_id: str, all_nodes: dict):
    entity_key = f"{node_id}_entity"
    type_key = f"{node_id}_type"
    return all_nodes.get(entity_key) or all_nodes.get(type_key)


def _populate_node_relations(all_nodes: dict, relationships: list) -> None:
    """Attach edges to nodes via .relations for downstream traversal and persistence."""
    for src_id, tgt_id, rel_name, properties in relationships:
        src_node = _resolve_node(src_id, all_nodes)
        tgt_node = _resolve_node(tgt_id, all_nodes)

        if src_node is None or tgt_node is None:
            continue

        src_node.relations.append(
            (
                Edge(
                    relationship_type=rel_name,
                    edge_text=(properties or {}).get("edge_text"),
                ),
                tgt_node,
            )
        )


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver=None,
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    """

    - LLM generated docstring
    Expand knowledge graphs with validated nodes and edges, integrating ontology information.

    This function processes document chunks and their associated knowledge graphs to create
    a comprehensive graph structure with entity nodes, entity type nodes, and their relationships.
    It validates entities against an ontology resolver and adds ontology-derived nodes and edges
    to enhance the knowledge representation.

    Args:
        data_chunks (list[DocumentChunk]): List of document chunks that contain the source data.
            Each chunk should have metadata about what entities it contains.
        chunk_graphs (list[KnowledgeGraph]): List of knowledge graphs corresponding to each
            data chunk. Each graph contains nodes (entities) and edges (relationships) extracted
            from the chunk content.
        ontology_resolver (BaseOntologyResolver, optional): Resolver for validating entities and
            types against an ontology. None means skip ontology enrichment.
        existing_edges_map (dict[str, bool], optional): Mapping of existing edge keys to prevent
            duplicate edge creation. Keys are formatted as "{source_id}_{target_id}_{relation}".
            If None, an empty dictionary is created. Defaults to None.

    Returns:
        tuple[list, list]: A tuple containing:
            - graph_nodes (list): Combined list of data chunks and ontology nodes (EntityType and Entity objects)
            - graph_edges (list): List of edge tuples in format (source_id, target_id, relationship_name, properties)

    Note:
        - Entity nodes are created for each entity found in the knowledge graphs
        - EntityType nodes are created for each unique entity type
        - Ontology validation is performed to map entities to canonical ontology terms
        - Duplicate nodes and edges are prevented using internal mapping and the existing_edges_map
        - The function modifies data_chunks in-place by adding entities to their 'contains' attribute

    """
    if existing_edges_map is None:
        existing_edges_map = {}

    chunk_graphs, matches = canonicalize_graphs(chunk_graphs, data_chunks, ontology_resolver)

    added_nodes_map = {}
    relationships = []

    for data_chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph:
            continue

        _process_graph_nodes(data_chunk, graph, added_nodes_map)
        _process_graph_edges(graph, existing_edges_map, relationships)

    added_ontology_nodes_map, ontology_relationships = extend_graph_with_ontology(
        matches, added_nodes_map, existing_edges_map
    )

    all_nodes = {**added_nodes_map, **added_ontology_nodes_map}
    all_relationships = relationships + ontology_relationships
    _populate_node_relations(all_nodes, all_relationships)

    entity_nodes = list(added_nodes_map.values()) + list(added_ontology_nodes_map.values())

    return data_chunks, entity_nodes
