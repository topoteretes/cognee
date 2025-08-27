import asyncio
from typing import Type, List
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models import DocumentChunk
from cognee.tasks.temporal_graph.models import EventList
from cognee.modules.engine.utils.generate_event_datapoint import generate_event_datapoint


async def extract_events_and_timestamps(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Extracts events and entities from a chunk of documents."""
    events = await asyncio.gather(
        *[LLMGateway.extract_event_graph(chunk.text, EventList) for chunk in data_chunks]
    )

    for data_chunk, event_list in zip(data_chunks, events):
        for event in event_list.events:
            event_datapoint = generate_event_datapoint(event)
            data_chunk.contains.append(event_datapoint)

    return data_chunks
