import asyncio
from typing import List

from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph.extract_content_graph import (
    extract_content_nodes_and_relationships,
)


async def extract_relationships_from_data(
    data_chunks: list[DocumentChunk], n_rounds: int
) -> List[DocumentChunk]:
    """Extracts and integrates potential nodes and relationships from document chunks using multi-round extraction."""
    chunk_results = await asyncio.gather(
        *[extract_content_nodes_and_relationships(chunk.text, n_rounds) for chunk in data_chunks]
    )

    # Update chunks with their potential nodes and relationships
    for chunk, (nodes, relationships) in zip(data_chunks, chunk_results):
        chunk.potential_nodes = nodes
        chunk.potential_relationships = relationships

    return data_chunks
