import json
import asyncio
from uuid import uuid5, NAMESPACE_OID
from datetime import datetime, timezone
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import DataPoint, get_vector_engine
from cognee.modules.data.extraction.knowledge_graph.extract_content_graph import extract_content_graph
from cognee.modules.chunking import DocumentChunk
from cognee.modules.graph.utils import generate_node_id, generate_node_name


class EntityNode(BaseModel):
    uuid: str
    name: str
    type: str
    description: str
    created_at: datetime
    updated_at: datetime

async def chunks_into_graph(data_chunks: list[DocumentChunk], graph_model: Type[BaseModel], collection_name: str):
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    vector_engine = get_vector_engine()
    graph_engine = await get_graph_engine()

    has_collection = await vector_engine.has_collection(collection_name)

    if not has_collection:
        await vector_engine.create_collection(collection_name, payload_schema = EntityNode)

    processed_nodes = {}
    type_node_edges = []
    entity_node_edges = []
    type_entity_edges = []

    for (chunk_index, chunk) in enumerate(data_chunks):
        chunk_graph = chunk_graphs[chunk_index]
        for node in chunk_graph.nodes:
            type_node_id = generate_node_id(node.type)
            entity_node_id = generate_node_id(node.id)

            if type_node_id not in processed_nodes:
                type_node_edges.append((str(chunk.chunk_id), type_node_id, "contains_entity_type"))
                processed_nodes[type_node_id] = True

            if entity_node_id not in processed_nodes:
                entity_node_edges.append((str(chunk.chunk_id), entity_node_id, "contains_entity"))
                type_entity_edges.append((entity_node_id, type_node_id, "is_entity_type"))
                processed_nodes[entity_node_id] = True

        graph_node_edges = [
            (edge.target_node_id, edge.source_node_id, edge.relationship_name) \
                        for edge in chunk_graph.edges
        ]

    existing_edges = await graph_engine.has_edges([
        *type_node_edges,
        *entity_node_edges,
        *type_entity_edges,
        *graph_node_edges,
    ])

    existing_edges_map = {}
    existing_nodes_map = {}

    for edge in existing_edges:
        existing_edges_map[edge[0] + edge[1] + edge[2]] = True
        existing_nodes_map[edge[0]] = True

    graph_nodes = []
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

            if node_id not in existing_nodes_map:
                node_data = dict(
                    uuid = node_id,
                    name = node_name,
                    type = node_name,
                    description = node.description,
                    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )

                graph_nodes.append((
                    node_id,
                    dict(
                        **node_data,
                        properties = json.dumps(node.properties),
                    )
                ))

                data_points.append(DataPoint[EntityNode](
                    id = str(uuid5(NAMESPACE_OID, node_id)),
                    payload = node_data,
                    embed_field = "name",
                ))

                existing_nodes_map[node_id] = True

            edge_key = str(chunk.chunk_id) + node_id + "contains_entity"

            if edge_key not in existing_edges_map:
                graph_edges.append((
                    str(chunk.chunk_id),
                    node_id,
                    "contains_entity",
                    dict(
                        relationship_name = "contains_entity",
                        source_node_id = str(chunk.chunk_id),
                        target_node_id = node_id,
                    ),
                ))

                # Add relationship between entity type and entity itself: "Jake is Person"
                graph_edges.append((
                    node_id,
                    type_node_id,
                    "is_entity_type",
                    dict(
                        relationship_name = "is_entity_type",
                        source_node_id = type_node_id,
                        target_node_id = node_id,
                    ),
                ))

                existing_edges_map[edge_key] = True

            if type_node_id not in existing_nodes_map:
                type_node_data = dict(
                    uuid = type_node_id,
                    name = type_node_name,
                    type = type_node_id,
                    description = type_node_name,
                    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )

                graph_nodes.append((type_node_id, dict(
                    **type_node_data,
                    properties = json.dumps(node.properties)
                )))

                data_points.append(DataPoint[EntityNode](
                    id = str(uuid5(NAMESPACE_OID, type_node_id)),
                    payload = type_node_data,
                    embed_field = "name",
                ))

                existing_nodes_map[type_node_id] = True

            edge_key = str(chunk.chunk_id) + type_node_id + "contains_entity_type"

            if edge_key not in existing_edges_map:
                graph_edges.append((
                    str(chunk.chunk_id),
                    type_node_id,
                    "contains_entity_type",
                    dict(
                        relationship_name = "contains_entity_type",
                        source_node_id = str(chunk.chunk_id),
                        target_node_id = type_node_id,
                    ),
                ))

                existing_edges_map[edge_key] = True

            # Add relationship that came from graphs.
            for edge in graph.edges:
                source_node_id = generate_node_id(edge.source_node_id)
                target_node_id = generate_node_id(edge.target_node_id)
                relationship_name = generate_node_name(edge.relationship_name)
                edge_key = source_node_id + target_node_id + relationship_name

                if edge_key not in existing_edges_map:
                    graph_edges.append((
                        generate_node_id(edge.source_node_id),
                        generate_node_id(edge.target_node_id),
                        edge.relationship_name,
                        dict(
                            relationship_name = generate_node_name(edge.relationship_name),
                            source_node_id = generate_node_id(edge.source_node_id),
                            target_node_id = generate_node_id(edge.target_node_id),
                            properties = json.dumps(edge.properties),
                        ),
                    ))
                    existing_edges_map[edge_key] = True

    if len(data_points) > 0:
        await vector_engine.create_data_points(collection_name, data_points)

    if len(graph_nodes) > 0:
        await graph_engine.add_nodes(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks
