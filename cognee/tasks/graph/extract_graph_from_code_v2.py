import asyncio
from typing import List

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.graph.cascade_extract.utils.extract_nodes import extract_nodes
from cognee.tasks.graph.cascade_extract.utils.extract_content_nodes_and_relationship_names import (
    extract_content_nodes_and_relationship_names,
)
from cognee.tasks.graph.cascade_extract.utils.extract_edge_triplets import (
    extract_edge_triplets,
)
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs


async def extract_graph_from_data(
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

    return await integrate_chunk_graphs(data_chunks, chunk_graphs, KnowledgeGraph)
