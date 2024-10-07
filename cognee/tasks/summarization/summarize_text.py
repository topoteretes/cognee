
import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.vector import get_vector_engine, DataPoint
from cognee.modules.data.extraction.extract_summary import extract_summary
from cognee.modules.chunking import DocumentChunk
from .models.TextSummary import TextSummary


async def summarize_text(data_chunks: list[DocumentChunk], summarization_model: Type[BaseModel], collection_name: str = "summaries"):
    if len(data_chunks) == 0:
        return data_chunks

    chunk_summaries = await asyncio.gather(
        *[extract_summary(chunk.text, summarization_model) for chunk in data_chunks]
    )

    vector_engine = get_vector_engine()

    await vector_engine.create_collection(collection_name, payload_schema=TextSummary)

    await vector_engine.create_data_points(
        collection_name,
        [
            DataPoint[TextSummary](
                id = str(chunk.chunk_id),
                payload = dict(
                    chunk_id = str(chunk.chunk_id),
                    document_id = str(chunk.document_id),
                    text = chunk_summaries[chunk_index].summary,
                ),
                embed_field = "text",
            ) for (chunk_index, chunk) in enumerate(data_chunks)
        ],
    )

    return data_chunks
