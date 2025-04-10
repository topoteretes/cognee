import asyncio
from typing import Type
from uuid import uuid5
from pydantic import BaseModel
from cognee.modules.data.extraction.extract_summary import extract_summary
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from .models import TextSummary


async def summarize_text(data_chunks: list[DocumentChunk], summarization_model: Type[BaseModel]):
    if len(data_chunks) == 0:
        return data_chunks

    chunk_summaries = await asyncio.gather(
        *[extract_summary(chunk.text, summarization_model) for chunk in data_chunks]
    )

    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            text=chunk_summaries[chunk_index].summary,
        )
        for (chunk_index, chunk) in enumerate(data_chunks)
    ]

    return summaries
