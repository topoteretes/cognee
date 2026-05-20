from typing import List, Type, Optional
from pydantic import BaseModel
import asyncio
from uuid import uuid5

from cognee.infrastructure.llm.extraction import extract_content_graph_and_summary
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.processing.document_types import DltRowDocument
from cognee.modules.ontology.ontology_config import Config
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.summarization.models import TextSummary


async def extract_graph_and_summarize(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    ctx=None,
    summarization_model: Type[BaseModel] = None,
    **kwargs,
) -> List[TextSummary]:
    calculate_chunk_graphs = kwargs.get("calculate_chunk_graphs")
    if callable(calculate_chunk_graphs):
        return await _extract_graph_and_summarize_separately(
            data_chunks=data_chunks,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            ctx=ctx,
            summarization_model=summarization_model,
            **kwargs,
        )

    non_dlt_chunks = [
        c for c in data_chunks if not isinstance(getattr(c, "is_part_of", None), DltRowDocument)
    ]
    dlt_chunks = [c for c in data_chunks if c not in non_dlt_chunks]

    if not non_dlt_chunks:
        return data_chunks

    if summarization_model is None:
        summarization_model = get_cognify_config().summarization_model

    chunk_extractions = await asyncio.gather(
        *[
            extract_content_graph_and_summary(
                chunk.text,
                graph_model=graph_model,
                summarization_model=summarization_model,
                custom_prompt=custom_prompt,
                **kwargs,
            )
            for chunk in non_dlt_chunks
        ]
    )

    chunk_graphs = [chunk_extraction.graph for chunk_extraction in chunk_extractions]
    chunk_summaries = [chunk_extraction.summary for chunk_extraction in chunk_extractions]

    async def _use_combined_chunk_graphs(*_args, **_kwargs):
        return chunk_graphs

    await extract_graph_from_data(
        data_chunks=data_chunks,
        graph_model=graph_model,
        config=config,
        custom_prompt=custom_prompt,
        ctx=ctx,
        calculate_chunk_graphs=_use_combined_chunk_graphs,
        **kwargs,
    )

    summaries = [
        TextSummary(
            id=uuid5(chunk.id, "TextSummary"),
            made_from=chunk,
            text=chunk_summaries[chunk_index].summary,
            importance_weight=chunk.importance_weight,
        )
        for (chunk_index, chunk) in enumerate(non_dlt_chunks)
    ]

    return summaries + dlt_chunks


async def _extract_graph_and_summarize_separately(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    ctx=None,
    summarization_model: Type[BaseModel] = None,
    **kwargs,
) -> List[TextSummary]:
    result_chunks = await asyncio.gather(
        extract_graph_from_data(
            data_chunks=data_chunks,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            ctx=ctx,
            **kwargs,
        ),
        summarize_text(
            data_chunks=data_chunks,
            summarization_model=summarization_model,
        ),
    )

    # Return only TextSummary objects, keeping the same logic as sequential execution of these tasks
    return result_chunks[1]
