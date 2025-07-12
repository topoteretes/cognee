from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver

# Constants for node key suffixes
TYPE_SUFFIX = "_type"
ENTITY_SUFFIX = "_entity"


def _create_ontology_node(ont_node, category: str, data_chunk: DocumentChunk):
    """Create an ontology node based on category (classes or individuals)."""
    ont_node_id = generate_node_id(ont_node.name)
    ont_node_name = generate_node_name(ont_node.name)

    if category == "classes":
        return f"{ont_node_id}{TYPE_SUFFIX}", EntityType(
            id=ont_node_id,
            name=ont_node_name,
            description=ont_node_name,
            ontology_valid=True,
        )
    elif category == "individuals":
        return f"{ont_node_id}{ENTITY_SUFFIX}", Entity(
            id=ont_node_id,
            name=ont_node_name,
            description=ont_node_name,
            ontology_valid=True,
            belongs_to_set=data_chunk.belongs_to_set,
        )
    return None, None


def _process_ontology_nodes(
    ontology_nodes, data_chunk: DocumentChunk, added_nodes_map: dict, added_ontology_nodes_map: dict
):
    """Process and add ontology nodes to the appropriate maps."""
    for ont_node in ontology_nodes:
        ont_node_key, ont_node_obj = _create_ontology_node(ont_node, ont_node.category, data_chunk)
        if (
            ont_node_key
            and ont_node_obj
            and ont_node_key not in added_nodes_map
            and ont_node_key not in added_ontology_nodes_map
        ):
            added_ontology_nodes_map[ont_node_key] = ont_node_obj


def _process_ontology_edges(
    ontology_edges,
    existing_edges_map: dict,
    ontology_relationships: list,
    ontology_valid: bool = True,
):
    """Process ontology edges and add them to relationships if not already existing."""
    for source, relation, target in ontology_edges:
        source_node_id = generate_node_id(source)
        target_node_id = generate_node_id(target)
        relationship_name = generate_edge_name(relation)
        edge_key = f"{source_node_id}_{target_node_id}_{relationship_name}"

        if edge_key not in existing_edges_map:
            ontology_relationships.append(
                (
                    source_node_id,
                    target_node_id,
                    relationship_name,
                    {
                        "relationship_name": relationship_name,
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "ontology_valid": ontology_valid,
                    },
                )
            )
            existing_edges_map[edge_key] = True


def _resolve_ontology_mapping(node_name: str, node_type: str, ontology_resolver: OntologyResolver):
    """Resolve ontology mapping for a node and return validation result and closest match."""
    ontology_nodes, ontology_edges, closest_node = ontology_resolver.get_subgraph(
        node_name=node_name, node_type=node_type
    )

    ontology_validated = bool(closest_node)
    mapped_name = closest_node.name if closest_node else node_name

    return ontology_nodes, ontology_edges, ontology_validated, mapped_name


def _get_or_create_type_node(
    node,
    data_chunk: DocumentChunk,
    ontology_resolver: OntologyResolver,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
    name_mapping: dict,
    key_mapping: dict,
    existing_edges_map: dict,
    ontology_relationships: list,
):
    """Get or create a type node with ontology validation."""
    type_node_id = generate_node_id(node.type)
    type_node_name = generate_node_name(node.type)
    type_node_key = f"{type_node_id}{TYPE_SUFFIX}"

    # Check if node already exists
    if type_node_key in added_nodes_map or type_node_key in key_mapping:
        return added_nodes_map.get(type_node_key) or added_nodes_map.get(
            key_mapping.get(type_node_key)
        )

    # Resolve ontology for type
    ontology_nodes, ontology_edges, ontology_validated, mapped_type_name = (
        _resolve_ontology_mapping(type_node_name, "classes", ontology_resolver)
    )

    # Update mappings if ontology validation succeeded
    if ontology_validated:
        name_mapping[type_node_name] = mapped_type_name
        old_key = type_node_key
        type_node_id = generate_node_id(mapped_type_name)
        type_node_key = f"{type_node_id}{TYPE_SUFFIX}"
        type_node_name = generate_node_name(mapped_type_name)
        key_mapping[old_key] = type_node_key

    # Create type node
    type_node = EntityType(
        id=type_node_id,
        name=type_node_name,
        type=type_node_name,
        description=type_node_name,
        ontology_valid=ontology_validated,
    )
    added_nodes_map[type_node_key] = type_node

    # Process ontology nodes and edges
    _process_ontology_nodes(ontology_nodes, data_chunk, added_nodes_map, added_ontology_nodes_map)
    _process_ontology_edges(ontology_edges, existing_edges_map, ontology_relationships)

    return type_node


def _get_or_create_entity_node(
    node,
    type_node,
    data_chunk: DocumentChunk,
    ontology_resolver: OntologyResolver,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
    name_mapping: dict,
    key_mapping: dict,
    existing_edges_map: dict,
    ontology_relationships: list,
):
    """Get or create an entity node with ontology validation."""
    node_id = generate_node_id(node.id)
    node_name = generate_node_name(node.name)
    entity_node_key = f"{node_id}{ENTITY_SUFFIX}"

    # Check if node already exists
    if entity_node_key in added_nodes_map or entity_node_key in key_mapping:
        return added_nodes_map.get(entity_node_key) or added_nodes_map.get(
            key_mapping.get(entity_node_key)
        )

    # Resolve ontology for entity
    ontology_nodes, ontology_edges, ontology_validated, mapped_entity_name = (
        _resolve_ontology_mapping(node_name, "individuals", ontology_resolver)
    )

    # Update mappings if ontology validation succeeded
    if ontology_validated:
        name_mapping[node_name] = mapped_entity_name
        old_key = entity_node_key
        node_id = generate_node_id(mapped_entity_name)
        entity_node_key = f"{node_id}{ENTITY_SUFFIX}"
        node_name = generate_node_name(mapped_entity_name)
        key_mapping[old_key] = entity_node_key

    # Create entity node
    entity_node = Entity(
        id=node_id,
        name=node_name,
        is_a=type_node,
        description=node.description,
        ontology_valid=ontology_validated,
        belongs_to_set=data_chunk.belongs_to_set,
    )
    added_nodes_map[entity_node_key] = entity_node

    # Process ontology nodes and edges
    _process_ontology_nodes(ontology_nodes, data_chunk, added_nodes_map, added_ontology_nodes_map)
    _process_ontology_edges(
        ontology_edges, existing_edges_map, ontology_relationships, ontology_valid=True
    )

    return entity_node


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: OntologyResolver = None,
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    """
    Expand chunk graphs with nodes and edges, applying ontology validation.

    Args:
        data_chunks: List of document chunks
        chunk_graphs: List of knowledge graphs corresponding to chunks
        ontology_resolver: Optional ontology resolver for validation
        existing_edges_map: Optional map of existing edges to avoid duplicates

    Returns:
        Tuple of (graph_nodes, graph_edges)
    """
    existing_edges_map = existing_edges_map or {}
    ontology_resolver = ontology_resolver or OntologyResolver()

    added_nodes_map = {}
    added_ontology_nodes_map = {}
    relationships = []
    ontology_relationships = []
    name_mapping = {}
    key_mapping = {}

    for data_chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph:
            continue

        # Process nodes
        for node in graph.nodes:
            # Get or create type node
            type_node = _get_or_create_type_node(
                node,
                data_chunk,
                ontology_resolver,
                added_nodes_map,
                added_ontology_nodes_map,
                name_mapping,
                key_mapping,
                existing_edges_map,
                ontology_relationships,
            )

            # Get or create entity node
            entity_node = _get_or_create_entity_node(
                node,
                type_node,
                data_chunk,
                ontology_resolver,
                added_nodes_map,
                added_ontology_nodes_map,
                name_mapping,
                key_mapping,
                existing_edges_map,
                ontology_relationships,
            )

            # Add entity to chunk
            if data_chunk.contains is None:
                data_chunk.contains = []
            data_chunk.contains.append(entity_node)

        # Process edges
        for edge in graph.edges:
            source_node_id = generate_node_id(
                name_mapping.get(edge.source_node_id, edge.source_node_id)
            )
            target_node_id = generate_node_id(
                name_mapping.get(edge.target_node_id, edge.target_node_id)
            )
            relationship_name = generate_edge_name(edge.relationship_name)
            edge_key = f"{source_node_id}_{target_node_id}_{relationship_name}"

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
                        },
                    )
                )
                existing_edges_map[edge_key] = True

    graph_nodes = list(added_ontology_nodes_map.values())
    graph_edges = relationships + ontology_relationships

    return graph_nodes, graph_edges
