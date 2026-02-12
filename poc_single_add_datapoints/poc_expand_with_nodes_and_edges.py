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
    if data_chunk.contains is None:
        data_chunk.contains = []

    # Filter data_chunk.contains Entities that exist in both, data_chunk.contains and graph_entity_nodes
    data_chunk.contains = [
        t for t in data_chunk.contains if t[1].id not in graph_entity_nodes.keys()
    ]

    for entity_node in graph_entity_nodes.values():
        if isinstance(entity_node, GraphEntityType):
            continue

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


def _convert_entities_to_graph_entities(added_nodes_map, added_ontology_nodes_map):
    graph_entity_nodes = {}
    for node in chain(added_nodes_map.values(), added_ontology_nodes_map.values()):
        if node.id in graph_entity_nodes:
            continue
        graph_node = _to_graph_entity(node)
        if graph_node is not None:
            graph_entity_nodes[node.id] = graph_node
    return graph_entity_nodes


def _populate_graph_entities_entity_relations(graph_entity_nodes, relationships):
    for edge in relationships:
        source_entity = graph_entity_nodes.get(edge[0])
        target_entity = graph_entity_nodes.get(edge[1])
        if source_entity is None or target_entity is None:
            continue
        source_entity.relations.append((Edge(relationship_type=edge[2]), target_entity))


def _build_graph_entities_with_relations(
    added_nodes_map, added_ontology_nodes_map, relationships, ontology_relationships
):
    # Create Graph Entity nodes from added_nodes_map and added_ontology_nodes_map
    graph_entity_nodes = _convert_entities_to_graph_entities(
        added_nodes_map, added_ontology_nodes_map
    )

    # Fill relations between each Graph Entity based off of relationships
    _populate_graph_entities_entity_relations(
        graph_entity_nodes, chain(relationships, ontology_relationships)
    )

    return graph_entity_nodes


def poc_expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    ontology_resolver: BaseOntologyResolver = None,
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    # region same as expand_with_nodes_and_edges
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
    # endregion
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

        # Transform newly created nodes into GraphEntities and populate with relations
        graph_entity_nodes = _build_graph_entities_with_relations(
            added_nodes_map, added_ontology_nodes_map, relationships, ontology_relationships
        )

        # Link Graph Entities to their respective data_chunk using contains and reset maps
        _link_graph_entities_to_data_chunk(data_chunk, graph_entity_nodes)

        # Reset maps to keep track of which nodes and relationships were added in current document chunk
        added_nodes_map.clear()
        added_ontology_nodes_map.clear()
        relationships.clear()
        ontology_relationships.clear()
