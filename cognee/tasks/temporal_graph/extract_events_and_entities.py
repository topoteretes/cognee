import asyncio
from typing import Type, List
from cognee.infrastructure.llm.extraction import extract_event_graph
from cognee.modules.chunking.models import DocumentChunk
from cognee.tasks.temporal_graph.models import EventList
from cognee.modules.engine.utils.generate_event_datapoint import generate_event_datapoint


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
    events = await asyncio.gather(
        *[extract_event_graph(chunk.text, EventList) for chunk in data_chunks]
    )

    for data_chunk, event_list in zip(data_chunks, events):
        for event in event_list.events:
            event_datapoint = generate_event_datapoint(event)
            data_chunk.contains.append(event_datapoint)

    return data_chunks
