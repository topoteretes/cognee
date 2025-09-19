from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)


def _create_node_key(node_id: str, category: str) -> str:
    """Create a standardized node key"""
    return f"{node_id}_{category}"


def _create_edge_key(source_id: str, target_id: str, relationship_name: str) -> str:
    """Create a standardized edge key"""
    return f"{source_id}_{target_id}_{relationship_name}"


def _process_ontology_nodes(
    ontology_nodes: list,
    data_chunk: DocumentChunk,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
) -> None:
    """Process and store ontology nodes"""
    for ontology_node in ontology_nodes:
        ont_node_id = generate_node_id(ontology_node.name)
        ont_node_name = generate_node_name(ontology_node.name)

        if ontology_node.category == "classes":
            ont_node_key = _create_node_key(ont_node_id, "type")
            if ont_node_key not in added_nodes_map and ont_node_key not in added_ontology_nodes_map:
                added_ontology_nodes_map[ont_node_key] = EntityType(
                    id=ont_node_id,
                    name=ont_node_name,
                    description=ont_node_name,
                    ontology_valid=True,
                )

        elif ontology_node.category == "individuals":
            ont_node_key = _create_node_key(ont_node_id, "entity")
            if ont_node_key not in added_nodes_map and ont_node_key not in added_ontology_nodes_map:
                added_ontology_nodes_map[ont_node_key] = Entity(
                    id=ont_node_id,
                    name=ont_node_name,
                    description=ont_node_name,
                    ontology_valid=True,
                    belongs_to_set=data_chunk.belongs_to_set,
                )


def _process_ontology_edges(
    ontology_edges: list, existing_edges_map: dict, ontology_relationships: list
) -> None:
    """Process ontology edges and add them if new"""
    for source, relation, target in ontology_edges:
        source_node_id = generate_node_id(source)
        target_node_id = generate_node_id(target)
        relationship_name = generate_edge_name(relation)
        edge_key = _create_edge_key(source_node_id, target_node_id, relationship_name)

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
                        "ontology_valid": True,
                    },
                )
            )
            existing_edges_map[edge_key] = True


def _create_type_node(
    node_type: str,
    ontology_resolver: RDFLibOntologyResolver,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
    name_mapping: dict,
    key_mapping: dict,
    data_chunk: DocumentChunk,
    existing_edges_map: dict,
    ontology_relationships: list,
) -> EntityType:
    """Create or retrieve a type node with ontology validation"""
    node_id = generate_node_id(node_type)
    node_name = generate_node_name(node_type)
    type_node_key = _create_node_key(node_id, "type")

    if type_node_key in added_nodes_map or type_node_key in key_mapping:
        return added_nodes_map.get(type_node_key) or added_nodes_map.get(
            key_mapping.get(type_node_key)
        )

    # Get ontology validation
    ontology_nodes, ontology_edges, closest_class = ontology_resolver.get_subgraph(
        node_name=node_name, node_type="classes"
    )

    ontology_validated = bool(closest_class)

    if ontology_validated:
        old_key = type_node_key
        node_id = generate_node_id(closest_class.name)
        type_node_key = _create_node_key(node_id, "type")
        new_node_name = generate_node_name(closest_class.name)

        name_mapping[node_name] = closest_class.name
        key_mapping[old_key] = type_node_key
        node_name = new_node_name

    type_node = EntityType(
        id=node_id,
        name=node_name,
        type=node_name,
        description=node_name,
        ontology_valid=ontology_validated,
    )

    added_nodes_map[type_node_key] = type_node

    # Process ontology nodes and edges
    _process_ontology_nodes(ontology_nodes, data_chunk, added_nodes_map, added_ontology_nodes_map)
    _process_ontology_edges(ontology_edges, existing_edges_map, ontology_relationships)

    return type_node


def _create_entity_node(
    node_id: str,
    node_name: str,
    node_description: str,
    type_node: EntityType,
    ontology_resolver: RDFLibOntologyResolver,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
    name_mapping: dict,
    key_mapping: dict,
    data_chunk: DocumentChunk,
    existing_edges_map: dict,
    ontology_relationships: list,
) -> Entity:
    """Create or retrieve an entity node with ontology validation"""
    generated_node_id = generate_node_id(node_id)
    generated_node_name = generate_node_name(node_name)
    entity_node_key = _create_node_key(generated_node_id, "entity")

    if entity_node_key in added_nodes_map or entity_node_key in key_mapping:
        return added_nodes_map.get(entity_node_key) or added_nodes_map.get(
            key_mapping.get(entity_node_key)
        )

    # Get ontology validation
    ontology_nodes, ontology_edges, start_ent_ont = ontology_resolver.get_subgraph(
        node_name=generated_node_name, node_type="individuals"
    )

    ontology_validated = bool(start_ent_ont)

    if ontology_validated:
        old_key = entity_node_key
        generated_node_id = generate_node_id(start_ent_ont.name)
        entity_node_key = _create_node_key(generated_node_id, "entity")
        new_node_name = generate_node_name(start_ent_ont.name)

        name_mapping[generated_node_name] = start_ent_ont.name
        key_mapping[old_key] = entity_node_key
        generated_node_name = new_node_name

    entity_node = Entity(
        id=generated_node_id,
        name=generated_node_name,
        is_a=type_node,
        description=node_description,
        ontology_valid=ontology_validated,
        belongs_to_set=data_chunk.belongs_to_set,
    )

    added_nodes_map[entity_node_key] = entity_node

    # Process ontology nodes and edges
    _process_ontology_nodes(ontology_nodes, data_chunk, added_nodes_map, added_ontology_nodes_map)
    _process_ontology_edges(ontology_edges, existing_edges_map, ontology_relationships)

    return entity_node


def _process_graph_nodes(
    data_chunk: DocumentChunk,
    graph: KnowledgeGraph,
    ontology_resolver: RDFLibOntologyResolver,
    added_nodes_map: dict,
    added_ontology_nodes_map: dict,
    name_mapping: dict,
    key_mapping: dict,
    existing_edges_map: dict,
    ontology_relationships: list,
) -> None:
    """Process nodes in a knowledge graph"""
    for node in graph.nodes:
        # Create type node
        type_node = _create_type_node(
            node.type,
            ontology_resolver,
            added_nodes_map,
            added_ontology_nodes_map,
            name_mapping,
            key_mapping,
            data_chunk,
            existing_edges_map,
            ontology_relationships,
        )

        # Create entity node
        entity_node = _create_entity_node(
            node.id,
            node.name,
            node.description,
            type_node,
            ontology_resolver,
            added_nodes_map,
            added_ontology_nodes_map,
            name_mapping,
            key_mapping,
            data_chunk,
            existing_edges_map,
            ontology_relationships,
        )

        # Add entity to data chunk
        if data_chunk.contains is None:
            data_chunk.contains = []
        data_chunk.contains.append(entity_node)


def _process_graph_edges(
    graph: KnowledgeGraph, name_mapping: dict, existing_edges_map: dict, relationships: list
) -> None:
    """Process edges in a knowledge graph"""
    for edge in graph.edges:
        # Apply name mapping if exists
        source_id = name_mapping.get(edge.source_node_id, edge.source_node_id)
        target_id = name_mapping.get(edge.target_node_id, edge.target_node_id)

        source_node_id = generate_node_id(source_id)
        target_node_id = generate_node_id(target_id)
        relationship_name = generate_edge_name(edge.relationship_name)
        edge_key = _create_edge_key(source_node_id, target_node_id, relationship_name)

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


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver = None,
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
            types against an ontology. If None, a default RDFLibOntologyResolver is created.
            Defaults to None.
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

    if ontology_resolver is None:
        ontology_config = get_ontology_env_config()
        if (
            ontology_config.ontology_file_path
            and ontology_config.ontology_resolver
            and ontology_config.matching_strategy
        ):
            ontology_resolver = get_ontology_resolver_from_env(**ontology_config.to_dict())
        else:
            ontology_resolver = get_default_ontology_resolver()

    added_nodes_map = {}
    added_ontology_nodes_map = {}
    relationships = []
    ontology_relationships = []
    name_mapping = {}
    key_mapping = {}

    # Process each chunk and its corresponding graph
    for data_chunk, graph in zip(data_chunks, chunk_graphs):
        if not graph:
            continue

        # Process nodes first
        _process_graph_nodes(
            data_chunk,
            graph,
            ontology_resolver,
            added_nodes_map,
            added_ontology_nodes_map,
            name_mapping,
            key_mapping,
            existing_edges_map,
            ontology_relationships,
        )

        # Then process edges
        _process_graph_edges(graph, name_mapping, existing_edges_map, relationships)

    # Return combined results
    graph_nodes = data_chunks + list(added_ontology_nodes_map.values())
    graph_edges = relationships + ontology_relationships

    return graph_nodes, graph_edges
