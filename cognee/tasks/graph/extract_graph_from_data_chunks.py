import asyncio
from typing import List

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.utils import expand_with_nodes_and_edges, retrieve_existing_edges
from cognee.tasks.storage import add_data_points
from cognee.tasks.graph.extract.extract_nodes import extract_nodes
from cognee.tasks.graph.extract.extract_relationship_names import (
    extract_content_nodes_and_relationship_names,
)
from cognee.tasks.graph.extract.extract_edge_triplets import extract_edge_triplets


async def extract_graph_from_data_chunks(
    data_chunks: List[DocumentChunk], n_rounds: int = 2
) -> List[DocumentChunk]:
    """Extract and update graph data from document chunks in multiple steps."""
    chunk_nodes = await asyncio.gather(
        *[extract_nodes(chunk.text, n_rounds) for chunk in data_chunks]
    )

    chunk_results = await asyncio.gather(
        *[
            extract_content_nodes_and_relationship_names(chunk.text, nodes, n_rounds)
            for chunk, nodes in zip(data_chunks, chunk_nodes)
        ]
    )

    updated_nodes, relationships = zip(*chunk_results)

    chunk_graphs = await asyncio.gather(
        *[
            extract_edge_triplets(chunk.text, nodes, rels, n_rounds)
            for chunk, nodes, rels in zip(data_chunks, updated_nodes, relationships)
        ]
    )

    graph_engine = await get_graph_engine()
    existing_edges_map = await retrieve_existing_edges(data_chunks, chunk_graphs, graph_engine)
    graph_nodes, graph_edge_triplets = expand_with_nodes_and_edges(
        data_chunks, chunk_graphs, existing_edges_map
    )

    if graph_nodes:
        await add_data_points(graph_nodes)
    if graph_edge_triplets:
        await graph_engine.add_edges(graph_edge_triplets)

    return data_chunks
