from typing import List
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Event
from cognee.tasks.temporal_graph.enrich_events import enrich_events
from cognee.tasks.temporal_graph.add_entities_to_event import add_entities_to_event


async def extract_knowledge_graph_from_events(
    data_chunks: List[DocumentChunk],
) -> List[DocumentChunk]:
    """
    Extracts events from document chunks and enriches them with entities to form a knowledge graph.

    The function collects all Event objects from the given document chunks,
    uses an LLM to extract and attach related entities, and updates the events
    with these enriched attributes.

    Args:
        data_chunks (List[DocumentChunk]): A list of document chunks containing extracted events.

    Returns:
        List[DocumentChunk]: The same list of document chunks, with their events enriched by entities.
    """
    # Extract events from chunks
    all_events = []
    for chunk in data_chunks:
        for item in chunk.contains:
            if isinstance(item, Event):
                all_events.append(item)

    if not all_events:
        return data_chunks

    # Enrich events with entities
    enriched_events = await enrich_events(all_events)

    # Add entities to events
    for event, enriched_event in zip(all_events, enriched_events):
        add_entities_to_event(event, enriched_event)

    return data_chunks
