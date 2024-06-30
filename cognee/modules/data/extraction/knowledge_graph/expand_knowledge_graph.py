import asyncio
from datetime import datetime
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from ...processing.chunk_types.DocumentChunk import DocumentChunk
from .extract_knowledge_graph import extract_content_graph

async def expand_knowledge_graph(data_chunks: list[DocumentChunk], graph_model: Type[BaseModel]):
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    graph_engine = await get_graph_engine()

    graph_nodes = []
    graph_edges = []

    for (chunk_index, chunk) in enumerate(data_chunks):
        graph = chunk_graphs[chunk_index]
        if graph is None:
            continue

        for node in graph.nodes:
            node_id = generate_node_id(node.id)

            for node in graph.nodes:
                graph_edges.append((
                    str(chunk.chunk_id),
                    node_id,
                    "contains",
                    dict(relationship_name = "contains"),
                ))

            type_node_id = generate_node_id(node.type)
            type_node = await graph_engine.extract_node(type_node_id)

            if not type_node:
                node_name = node.type.lower().capitalize()

                type_node = (
                    type_node_id,
                    dict(
                        id = type_node_id,
                        name = node_name,
                        type = node_name,
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                )

                graph_nodes.append(type_node)

            # Add relationship between entity type and entity itself: "Jake is Person"
            graph_edges.append((
                node_id,
                type_node_id,
                "is",
                dict(relationship_name = "is"),
            ))

            graph_nodes.append((
                node_id,
                dict(
                    id = node_id,
                    chunk_id = chunk.chunk_id,
                    document_id = chunk.document_id,
                    name = node.name,
                    type = node.type.lower().capitalize(),
                    description = node.description,
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            ))

            # Add relationship that came from graphs.
            for edge in graph.edges:
                graph_edges.append((
                    generate_node_id(edge.source_node_id),
                    generate_node_id(edge.target_node_id),
                    edge.relationship_name,
                    dict(relationship_name = edge.relationship_name),
                ))

    await graph_engine.add_nodes(graph_nodes)

    await graph_engine.add_edges(graph_edges)

    return data_chunks


def generate_node_id(node_id: str) -> str:
    return node_id.upper().replace(" ", "_").replace("'", "")
