import asyncio
from typing import Type
from uuid import uuid5
from pydantic import BaseModel

from cognee.tasks.summarization.exceptions import InvalidSummaryInputsError
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.infrastructure.llm.extraction import extract_summary
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.cognify.config import get_cognify_config
from cognee.tasks.summarization.models import TextSummary


from cognee.modules.pipelines.tasks.task import task_summary


class TemporalSummaryContent(BaseModel):
    text: str


async def extract_temporal_summary(content: str) -> TemporalSummaryContent:
    system_prompt = read_query_prompt("summarize_temporal_content.txt")
    return await LLMGateway.acreate_structured_output(content, system_prompt, TemporalSummaryContent)


async def _build_summary_text_from_chunk(
    chunk: DocumentChunk, summarization_model: Type[BaseModel]
) -> str:
    summary_result, temporal_result = await asyncio.gather(
        extract_summary(chunk.text, summarization_model),
        extract_temporal_summary(chunk.text),
    )

    summary_text = summary_result.summary.strip()
    temporal_text = temporal_result.text.strip()
    if not temporal_text:
        return f"Summary:\n{summary_text}"

    return f"Summary:\n{summary_text}\n\nEvents:\n{temporal_text}"


@task_summary("Summarized {n} chunk(s)")
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

    if not isinstance(data_chunks, list):
        raise InvalidSummaryInputsError("data_chunks must be a list.")
    if not all(hasattr(c, "text") for c in data_chunks):
        raise InvalidSummaryInputsError("each DocumentChunk must have a 'text' attribute.")

    if len(data_chunks) == 0:
        return data_chunks

    # Skip LLM summarization for DLT row chunks — structured data
    # doesn't benefit from text summarization.
    from cognee.modules.data.processing.document_types import DltRowDocument

    non_dlt_chunks = [
        c for c in data_chunks if not isinstance(getattr(c, "is_part_of", None), DltRowDocument)
    ]
    dlt_chunks = [c for c in data_chunks if c not in non_dlt_chunks]

    if not non_dlt_chunks:
        return data_chunks

    if summarization_model is None:
        cognee_config = get_cognify_config()
        summarization_model = cognee_config.summarization_model

    chunk_summaries = await asyncio.gather(
        *[_build_summary_text_from_chunk(chunk, summarization_model) for chunk in non_dlt_chunks]
    )

    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            text=chunk_summaries[chunk_index],
            importance_weight=chunk.importance_weight,
        )
        for (chunk_index, chunk) in enumerate(non_dlt_chunks)
    ]

    return summaries + dlt_chunks
