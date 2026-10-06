import asyncio
from typing import Literal

from pydantic import BaseModel

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.ontology.ontology_config import Config
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.graph.classify_entity_types import classify_chunk_entity_types
from cognee.tasks.summarization import summarize_text
from cognee.tasks.summarization.build_summary_from_extraction import build_summary_from_extraction
from cognee.tasks.summarization.models import TextSummary


async def extract_graph_and_summarize(
    data_chunks: list[DocumentChunk],
    graph_model: type[BaseModel],
    config: Config | None = None,
    custom_prompt: str | None = None,
    ctx=None,
    summarization_model: type[BaseModel] | None = None,
    chunk_attachment: Literal["direct", "all"] | None = None,
    summary_method: Literal["llm", "from_extraction"] | None = None,
    **kwargs,
) -> list[TextSummary | DocumentChunk]:
    cognify_config = get_cognify_config()
    if summary_method is None:
        summary_method = cognify_config.summary_method
    if summary_method == "from_extraction":
        # These summaries are built from what extraction returns, so extraction runs first.
        extracted_chunks = await extract_graph_from_data(
            data_chunks=data_chunks,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            ctx=ctx,
            chunk_attachment=chunk_attachment,
            **kwargs,
        )
        if cognify_config.entity_type_classification:
            await classify_chunk_entity_types(extracted_chunks)
        # A chunk with no relations has no summary. It is returned as itself so
        # add_data_points still stores it.
        return [
            await build_summary_from_extraction(chunk, graph_model) or chunk
            for chunk in extracted_chunks
        ]

    result_chunks = await asyncio.gather(
        extract_graph_from_data(
            data_chunks=data_chunks,
            graph_model=graph_model,
            config=config,
            custom_prompt=custom_prompt,
            ctx=ctx,
            chunk_attachment=chunk_attachment,
            **kwargs,
        ),
        summarize_text(
            data_chunks=data_chunks,
            summarization_model=summarization_model,
        ),
    )

    if cognify_config.entity_type_classification:
        await classify_chunk_entity_types(result_chunks[0])

    # Return only TextSummary objects, keeping the same logic as sequential execution of these tasks
    return result_chunks[1]
