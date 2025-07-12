import asyncio
from typing import Type
from uuid import uuid5
from pydantic import BaseModel

from cognee.base_config import get_base_config
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from .models import TextSummary

# Framework selection
base = get_base_config()
if base.structured_output_framework == "BAML":
    print(f"Using BAML framework for text summarization: {base.structured_output_framework}")
    from cognee.infrastructure.llm.structured_output_framework.baml_src.extraction import (
        extract_summary,
    )
else:
    print(
        f"Using llitellm_instructor framework for text summarization: {base.structured_output_framework}"
    )
    from cognee.infrastructure.llm.structured_output_framework.llitellm_instructor.extraction import (
        extract_summary,
    )


async def summarize_text(
    data_chunks: list[DocumentChunk], summarization_model: Type[BaseModel] = None
):
    """
    Summarize the text contained in the provided data chunks.

    If no summarization model is provided, the function retrieves the default model from the
    configuration. It processes the data chunks asynchronously and returns summaries for
    each chunk. If the provided list of data chunks is empty, it simply returns the list as
    is.

    Parameters:
    -----------

        - data_chunks (list[DocumentChunk]): A list of DocumentChunk objects containing text
          to be summarized.
        - summarization_model (Type[BaseModel]): An optional model used for summarizing
          text. If not provided, the default is fetched from the configuration. (default
          None)

    Returns:
    --------

        A list of TextSummary objects, each containing the summary of a corresponding
        DocumentChunk.
    """
    if len(data_chunks) == 0:
        return data_chunks

    if summarization_model is None:
        cognee_config = get_cognify_config()
        summarization_model = cognee_config.summarization_model

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
