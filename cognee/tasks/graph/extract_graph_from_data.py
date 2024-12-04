import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import EntityType, Entity
from cognee.modules.engine.utils import generate_edge_name, generate_node_id, generate_node_name
from cognee.tasks.storage import add_data_points

async def extract_graph_from_data(data_chunks: list[DocumentChunk], graph_model: Type[BaseModel]):
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    processed_nodes = {}
    type_node_edges = []
    entity_node_edges = []
    type_entity_edges = []

    for (chunk_index, chunk) in enumerate(data_chunks):
        chunk_graph = chunk_graphs[chunk_index]
        for node in chunk_graph.nodes:
            type_node_id = generate_node_id(node.type)
            entity_node_id = generate_node_id(node.id)

            if str(type_node_id) not in processed_nodes:
                type_node_edges.append((str(chunk.id), str(type_node_id), "exists_in"))
                processed_nodes[str(type_node_id)] = True

            if str(entity_node_id) not in processed_nodes:
                entity_node_edges.append((str(chunk.id), entity_node_id, "mentioned_in"))
                type_entity_edges.append((str(entity_node_id), str(type_node_id), "is_a"))
                processed_nodes[str(entity_node_id)] = True

        graph_node_edges = [
            (edge.target_node_id, edge.source_node_id, edge.relationship_name) \
                        for edge in chunk_graph.edges
        ]

    graph_engine = await get_graph_engine()

    existing_edges = await graph_engine.has_edges([
        *type_node_edges,
        *entity_node_edges,
        *type_entity_edges,
        *graph_node_edges,
    ])

    existing_edges_map = {}

    for edge in existing_edges:
        existing_edges_map[edge[0] + edge[1] + edge[2]] = True

    added_nodes_map = {}
    graph_edges = []
    data_points = []

    for (chunk_index, chunk) in enumerate(data_chunks):
        graph = chunk_graphs[chunk_index]
        if graph is None:
            continue

        for node in graph.nodes:
            node_id = generate_node_id(node.id)
            node_name = generate_node_name(node.name)

            type_node_id = generate_node_id(node.type)
            type_node_name = generate_node_name(node.type)

            if f"{str(type_node_id)}_type" not in added_nodes_map:
                type_node = EntityType(
                    id = type_node_id,
                    name = type_node_name,
                    type = type_node_name,
                    description = type_node_name,
                    exists_in = chunk,
                )
                added_nodes_map[f"{str(type_node_id)}_type"] = type_node
            else:
                type_node = added_nodes_map[f"{str(type_node_id)}_type"]

            if f"{str(node_id)}_entity" not in added_nodes_map:
                entity_node = Entity(
                    id = node_id,
                    name = node_name,
                    is_a = type_node,
                    description = node.description,
                    mentioned_in = chunk,
                )
                data_points.append(entity_node)
                added_nodes_map[f"{str(node_id)}_entity"] = entity_node

        # Add relationship that came from graphs.
        for edge in graph.edges:
            source_node_id = generate_node_id(edge.source_node_id)
            target_node_id = generate_node_id(edge.target_node_id)
            relationship_name = generate_edge_name(edge.relationship_name)

            edge_key = str(source_node_id) + str(target_node_id) + relationship_name

            if edge_key not in existing_edges_map:
                graph_edges.append((
                    source_node_id,
                    target_node_id,
                    edge.relationship_name,
                    dict(
                        relationship_name = generate_edge_name(edge.relationship_name),
                        source_node_id = source_node_id,
                        target_node_id = target_node_id,
                    ),
                ))
                existing_edges_map[edge_key] = True

    if len(data_points) > 0:
        await add_data_points(data_points)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks
