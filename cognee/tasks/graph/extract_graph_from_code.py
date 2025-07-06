import asyncio
from typing import Type, List
from pydantic import BaseModel
from cognee.infrastructure.llm.structured_output_framework.llitellm_instructor.extraction import extract_content_graph
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.tasks.storage import add_data_points
from cognee.base_config import get_base_config

base = get_base_config()

if base.structured_output_framework == 'BAML':
    from cognee.infrastructure.llm.structured_output_framework.baml_src.extraction import extract_content_graph
else:
    from cognee.infrastructure.llm.structured_output_framework.llitellm_instructor.extraction import extract_content_graph

async def extract_graph_from_code(
    data_chunks: list[DocumentChunk], graph_model: Type[BaseModel]
) -> List[DocumentChunk]:
    """
    Extracts a knowledge graph from the text content of document chunks using a specified graph model.

    Notes:
        - The `extract_content_graph` function processes each chunk's text to extract graph information.
        - Graph nodes are stored using the `add_data_points` function for later retrieval or analysis.
    """
    chunk_graphs = await asyncio.gather(
        *[extract_content_graph(chunk.text, graph_model) for chunk in data_chunks]
    )

    for chunk_index, chunk in enumerate(data_chunks):
        chunk_graph = chunk_graphs[chunk_index]
        await add_data_points(chunk_graph.nodes)

    return data_chunks
