import asyncio
from typing import List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph.extract_content_graph import (
    extract_content_edges,
)
from cognee.modules.graph.utils import (
    expand_with_nodes_and_edges,
    retrieve_existing_edges,
)
from cognee.tasks.storage import add_data_points


async def extract_edges_from_data(data_chunks: list[DocumentChunk]) -> List[DocumentChunk]:
    """Extracts and integrates edges between nodes using potential nodes and relationships from chunks."""
    chunk_graphs = await asyncio.gather(
        *[
            extract_content_edges(
                chunk.text, chunk.potential_nodes or [], chunk.potential_relationships or []
            )
            for chunk in data_chunks
        ]
    )
    graph_engine = await get_graph_engine()

    existing_edges_map = await retrieve_existing_edges(
        data_chunks,
        chunk_graphs,
        graph_engine,
    )

    graph_nodes, graph_edges = expand_with_nodes_and_edges(
        data_chunks,
        chunk_graphs,
        existing_edges_map,
    )

    if len(graph_nodes) > 0:
        await add_data_points(graph_nodes)

    if len(graph_edges) > 0:
        await graph_engine.add_edges(graph_edges)

    return data_chunks
