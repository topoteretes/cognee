import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.modules.data.extraction.knowledge_graph import extract_content_graph
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.tasks.storage import add_data_points


async def extract_graph_from_code(data_chunks: list[DocumentChunk], graph_model: Type[BaseModel]):
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    for chunk_index, chunk in enumerate(data_chunks):
        chunk_graph = chunk_graphs[chunk_index]
        await add_data_points(chunk_graph.nodes)

    return data_chunks
