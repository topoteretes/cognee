import asyncio
from typing import List

from cognee.infrastructure.llm.extraction import extract_event_graph
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Event
from cognee.modules.engine.utils.generate_event_datapoint import generate_event_datapoint
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.tasks.temporal_graph.models import EventList


async def extract_events_for_chunks(data_chunks: List[DocumentChunk]) -> List[List[Event]]:
    """Run the event extractor over every chunk concurrently, one LLM call per chunk.

    Pure with respect to the chunks: nothing is attached. Returns one list of
    ``Event`` datapoints per chunk, in chunk order, so a caller can run this
    alongside other per-chunk stages and attach the results afterwards.
    """
    event_lists = await asyncio.gather(
        *[extract_event_graph(chunk.text, EventList) for chunk in data_chunks]
    )
    return [
        [generate_event_datapoint(event) for event in event_list.events]
        for event_list in event_lists
    ]


def attach_events_to_chunks(
    data_chunks: List[DocumentChunk], events_per_chunk: List[List[Event]]
) -> List[DocumentChunk]:
    """Append each chunk's events to its ``contains`` list, keeping what is already there."""
    for data_chunk, events in zip(data_chunks, events_per_chunk):
        if not events:
            continue
        if data_chunk.contains is None:
            data_chunk.contains = []
        data_chunk.contains.extend(events)
    return data_chunks


@task_summary("Extracted events from {n} chunk(s)")
async def extract_events_and_timestamps(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Extracts events and their timestamps from document chunks using an LLM.

    Each document chunk is processed with the event graph extractor to identify events.
    The extracted events are converted into Event datapoints and appended to the
    chunk's `contains` list.

    Args:
        data_chunks (List[DocumentChunk]): A list of document chunks containing text to process.

    Returns:
        List[DocumentChunk]: The same list of document chunks, enriched with extracted Event datapoints.
    """
    return attach_events_to_chunks(data_chunks, await extract_events_for_chunks(data_chunks))
