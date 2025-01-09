from typing import Optional

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.engine.utils import (
    generate_edge_name,
    generate_node_id,
    generate_node_name,
)
from cognee.shared.data_models import KnowledgeGraph


def expand_with_nodes_and_edges(
    data_chunks: list[DocumentChunk],
    chunk_graphs: list[KnowledgeGraph],
    existing_edges_map: Optional[dict[str, bool]] = None,
):
    if existing_edges_map is None:
        existing_edges_map = {}

    added_nodes_map = {}
    relationships = []

    for index, data_chunk in enumerate(data_chunks):
        graph = chunk_graphs[index]

        if graph is None:
            continue

        for node in graph.nodes:
            node_id = generate_node_id(node.id)
            node_name = generate_node_name(node.name)

            type_node_id = generate_node_id(node.type)
            type_node_name = generate_node_name(node.type)

            if f"{str(type_node_id)}_type" not in added_nodes_map:
                type_node = EntityType(
                    id=type_node_id,
                    name=type_node_name,
                    type=type_node_name,
                    description=type_node_name,
                )
                added_nodes_map[f"{str(type_node_id)}_type"] = type_node
            else:
                type_node = added_nodes_map[f"{str(type_node_id)}_type"]

            if f"{str(node_id)}_entity" not in added_nodes_map:
                entity_node = Entity(
                    id=node_id,
                    name=node_name,
                    is_a=type_node,
                    description=node.description,
                )

                if data_chunk.contains is None:
                    data_chunk.contains = []

                data_chunk.contains.append(entity_node)

                added_nodes_map[f"{str(node_id)}_entity"] = entity_node

        # Add relationship that came from graphs.
        for edge in graph.edges:
            source_node_id = generate_node_id(edge.source_node_id)
            target_node_id = generate_node_id(edge.target_node_id)
            relationship_name = generate_edge_name(edge.relationship_name)

            edge_key = str(source_node_id) + str(target_node_id) + relationship_name

            if edge_key not in existing_edges_map:
                relationships.append(
                    (
                        source_node_id,
                        target_node_id,
                        edge.relationship_name,
                        dict(
                            relationship_name=generate_edge_name(edge.relationship_name),
                            source_node_id=source_node_id,
                            target_node_id=target_node_id,
                        ),
                    )
                )
                existing_edges_map[edge_key] = True

    return (data_chunks, relationships)
