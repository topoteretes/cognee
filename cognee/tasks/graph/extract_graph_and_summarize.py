import asyncio
from typing import Literal

from pydantic import BaseModel

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.ontology.ontology_config import Config
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.summarization.models import TextSummary


async def extract_graph_and_summarize(
    data_chunks: list[DocumentChunk],
    graph_model: type[BaseModel],
    config: Config | None = None,
    custom_prompt: str | None = None,
    ctx=None,
    summarization_model: type[BaseModel] | None = None,
    chunk_attachment: Literal["direct", "all"] | None = None,
    **kwargs,
) -> list[TextSummary]:
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

    # Return only TextSummary objects, keeping the same logic as sequential execution of these tasks
    return result_chunks[1]
