from typing import List, Literal, Type, Optional
from pydantic import BaseModel
import asyncio

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.ontology.ontology_config import Config
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.summarization import summarize_text
from cognee.tasks.summarization.models import TextSummary
from cognee.tasks.temporal_graph.extract_events_and_entities import (
    attach_events_to_chunks,
    extract_events_for_chunks,
)
from cognee.tasks.temporal_graph.link_events_to_entities import link_events_to_entities


async def extract_graph_and_summarize(
    data_chunks: List[DocumentChunk],
    graph_model: Type[BaseModel],
    config: Optional[Config] = None,
    custom_prompt: Optional[str] = None,
    ctx=None,
    summarization_model: Type[BaseModel] = None,
    chunk_attachment: Optional[Literal["direct", "all"]] = None,
    extract_events: bool = False,
    **kwargs,
) -> List[TextSummary]:
    """Run the per-chunk LLM stages concurrently and return the chunk summaries.

    ``extract_events=True`` (``cognify(temporal_cognify=True)``) adds the temporal
    event extraction as a third concurrent lane. Its events are attached to the
    chunks only after the graph extraction has written its entities — that stage
    assigns ``chunk.contains`` outright — and are then linked to the entities of the
    same chunk, so the timeline is layered on the knowledge graph rather than
    replacing it.
    """
    lanes = [
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
    ]
    if extract_events:
        lanes.append(extract_events_for_chunks(data_chunks))

    results = await asyncio.gather(*lanes)

    if extract_events:
        attach_events_to_chunks(data_chunks, results[2])
        link_events_to_entities(data_chunks)

    # Return only TextSummary objects, keeping the same logic as sequential execution of these tasks
    return results[1]
