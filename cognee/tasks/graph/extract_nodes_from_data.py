import asyncio
from typing import List

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.extraction.knowledge_graph.extract_content_graph import (
    extract_content_nodes,
    EntityListResponse,
)
from cognee.tasks.storage import add_data_points


async def extract_nodes_from_data(
    data_chunks: list[DocumentChunk], n_rounds: int
) -> List[DocumentChunk]:
    """Extracts and integrates entities from document chunks using multi-round entity extraction."""
    chunk_entities = await asyncio.gather(
        *[extract_content_nodes(chunk.text, n_rounds) for chunk in data_chunks]
    )

    # Update chunks with their entities and collect all entities for storage
    for chunk, entities in zip(data_chunks, chunk_entities):
        chunk.contains = entities

    return data_chunks
