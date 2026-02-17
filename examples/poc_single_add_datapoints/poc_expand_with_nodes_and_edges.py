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

        edge_text = "; ".join(
            [
                "relationship_name: contains",
                f"entity_name: {entity_node.name}",
                f"entity_description: {entity_node.description}",
            ]
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


def _replace_is_a_with_graph_nodes(data_chunk, graph_entity_nodes):
    if data_chunk.contains is None:
        data_chunk.contains = []

    for entity_node in data_chunk.contains:
        if entity_node[1].is_a.id in graph_entity_nodes:
            entity_node[1].is_a = graph_entity_nodes[entity_node[1].is_a.id]


def _to_graph_entity(node):
    if isinstance(node, GraphEntity) or isinstance(node, GraphEntityType):
        return node
    if isinstance(node, Entity):
        return GraphEntity(**node.model_dump(), relations=getattr(node, "relations", []))
    if isinstance(node, EntityType):
        return GraphEntityType(**node.model_dump(), relations=getattr(node, "relations", []))
    return None


def _convert_entities_to_graph_entities(added_ontology_nodes_map):
    graph_entity_nodes = {}
    for node in added_ontology_nodes_map.values():
        if node.id in graph_entity_nodes:
            continue
        graph_node = _to_graph_entity(node)
        if graph_node is not None:
            graph_entity_nodes[node.id] = graph_node
    return graph_entity_nodes


def _find_entity_by_id(node_id, data_chunks, graph_entity_nodes):
    if node_id in graph_entity_nodes:
        return graph_entity_nodes.get(node_id)
    for data_chunk in data_chunks:
        result = next((t for t in data_chunk.contains if t[1].id == node_id), None)
        if result:
            new_entity = _to_graph_entity(result[1])
            graph_entity_nodes[node_id] = new_entity
            return new_entity
        result = next((t for t in data_chunk.contains if t[1].is_a.id == node_id), None)
        if result:
            new_entity = _to_graph_entity(result[1].is_a)
            graph_entity_nodes[node_id] = new_entity
            result[1].is_a = new_entity
            return new_entity

    return None


def _populate_graph_entities_entity_relations(data_chunks, graph_entity_nodes, relationships):
    for edge in relationships:
        source_entity = _find_entity_by_id(edge[0], data_chunks, graph_entity_nodes)
        target_entity = _find_entity_by_id(edge[1], data_chunks, graph_entity_nodes)
        if source_entity is None or target_entity is None:
            continue
        source_entity.relations.append((Edge(relationship_type=edge[2]), target_entity))


def _build_graph_entities_with_relations(
    data_chunks, added_ontology_nodes_map, relationships, ontology_relationships
):
    # Create Graph Entity nodes from added_nodes_map and added_ontology_nodes_map
    graph_entity_nodes = _convert_entities_to_graph_entities(added_ontology_nodes_map)

    # Fill relations between each Graph Entity based off of relationships
    _populate_graph_entities_entity_relations(
        data_chunks, graph_entity_nodes, chain(relationships, ontology_relationships)
    )

    return graph_entity_nodes


def _filter_new_ontology_nodes(added_ontology_nodes_map_old, added_ontology_nodes_map):
    return {
        k: v for k, v in added_ontology_nodes_map.items() if k not in added_ontology_nodes_map_old
    }


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

        added_ontology_nodes_map_old = set(added_ontology_nodes_map.keys())

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

        new_added_ontology_nodes_map = _filter_new_ontology_nodes(
            added_ontology_nodes_map_old, added_ontology_nodes_map
        )

        # Transform newly created nodes into GraphEntities and populate with relations
        graph_entity_nodes = _build_graph_entities_with_relations(
            data_chunks, new_added_ontology_nodes_map, relationships, ontology_relationships
        )

        # Link Graph Entities to their respective data_chunk using contains and reset maps
        _link_graph_entities_to_data_chunk(data_chunk, graph_entity_nodes)
        _replace_is_a_with_graph_nodes(data_chunk, graph_entity_nodes)

        # Reset per-chunk edge collections only; keep node maps for cross-chunk deduplication
        relationships.clear()
        ontology_relationships.clear()
