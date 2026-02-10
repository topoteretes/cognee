from typing import Optional, Any

from pydantic import SkipValidation, Field

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.ontology_env_config import get_ontology_env_config
from cognee.shared.data_models import KnowledgeGraph
from cognee.modules.ontology.get_default_ontology_resolver import (
    get_default_ontology_resolver,
    get_ontology_resolver_from_env,
)
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    _process_graph_nodes,
    _process_graph_edges,
)

# from cognee.shared.data_models import Edge
from cognee.infrastructure.engine import Edge
from itertools import chain


class GraphEntity(Entity):
    relations: SkipValidation[Any] = Field(default_factory=list)


class GraphEntityType(EntityType):
    relations: SkipValidation[Any] = Field(default_factory=list)


def _link_graph_entities_to_data_chunk(data_chunk, graph_entity_nodes):
    data_chunk.contains = [
        t for t in data_chunk.contains if t[1].id not in graph_entity_nodes.keys()
    ]
    for entity_node in graph_entity_nodes.values():
        if entity_node.type == "GraphEntityType":
            continue

        if data_chunk.contains is None:
            data_chunk.contains = []

        data_chunk.contains.append(
            (
                Edge(relationship_type="contains"),
                entity_node,
            )
        )


def _to_graph_entity(node):
    if isinstance(node, Entity):
        return GraphEntity(**node.model_dump(), relations=getattr(node, "relations", []))
    if isinstance(node, EntityType):
        return GraphEntityType(**node.model_dump(), relations=getattr(node, "relations", []))
    return None


def convert_nodes_to_graph_entities(added_nodes_map, added_ontology_nodes_map):
    graph_entity_nodes = {}
    for node in chain(added_nodes_map.values(), added_ontology_nodes_map.values()):
        if node.id in graph_entity_nodes:
            continue
        graph_node = _to_graph_entity(node)
        if graph_node is not None:
            graph_entity_nodes[node.id] = graph_node
    return graph_entity_nodes


def populate_relations_from_ontology(
    added_nodes_map, added_ontology_nodes_map, relationships, ontology_relationships
):
    # Create Graph Entity nodes from added_nodes_map and added_ontology_nodes_map
    graph_entity_nodes = convert_nodes_to_graph_entities(added_nodes_map, added_ontology_nodes_map)

    # Fill relations between each Graph Entity based off of relationships
    for edge in chain(relationships, ontology_relationships):
        source_node_id, target_node_id = edge[0], edge[1]
        source_entity = graph_entity_nodes.get(source_node_id)
        target_entity = graph_entity_nodes.get(target_node_id)
        if source_entity is None or target_entity is None:
            continue

        source_entity.relations.append((Edge(relationship_type=edge[2]), target_entity))
    # graph_entity.relations.append((source_node_id, str(target_node_id), edge[2],
    #                                Edge(source_node_id=str(source_node_id), target_node_id=str(target_node_id),
    #                                     relationship_name=edge[2])))
    return graph_entity_nodes


def poc_expand_with_nodes_and_edges(
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

        graph_entity_nodes = populate_relations_from_ontology(
            added_nodes_map, added_ontology_nodes_map, relationships, ontology_relationships
        )

        # Link Graph Entities to their respective data_chunk using contains and reset maps
        _link_graph_entities_to_data_chunk(data_chunk, graph_entity_nodes)
        added_ontology_nodes_map.clear()
        ontology_relationships.clear()
