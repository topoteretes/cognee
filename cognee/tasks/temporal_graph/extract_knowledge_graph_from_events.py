from typing import Any, List
from cognee.modules.engine.models import Event
from cognee.tasks.temporal_graph.enrich_events import enrich_events
from cognee.tasks.temporal_graph.add_entities_to_event import add_entities_to_event


from cognee.modules.pipelines.tasks.task import task_summary


@task_summary("Built graph from {n} event(s)")
async def extract_knowledge_graph_from_events(
    data_chunks: List[Any],
) -> List[Any]:
    """
    Extracts events from document chunks and enriches them with entities to form a knowledge graph.

    The function collects all Event objects from the given document chunks,
    uses an LLM to extract and attach related entities, and updates the events
    with these enriched attributes.

    Args:
        data_chunks: Document chunks (or TextSummary items wrapping them) containing
            extracted events.

    Returns:
        The same list, with the events enriched by entities.
    """
    # Extract events from chunks. In the layered pipeline this task runs after
    # extract_graph_and_summarize and receives TextSummary items, which wrap their
    # source chunk in ``made_from``; bare chunks are accepted too.
    all_events = []
    for item in data_chunks:
        chunk = getattr(item, "made_from", None) or item
        for entry in getattr(chunk, "contains", None) or []:
            if isinstance(entry, Event):
                all_events.append(entry)

    if not all_events:
        return data_chunks

    # Enrich events with entities
    enriched_events = await enrich_events(all_events)

    # Add entities to events
    for event, enriched_event in zip(all_events, enriched_events):
        add_entities_to_event(event, enriched_event)

    return data_chunks
