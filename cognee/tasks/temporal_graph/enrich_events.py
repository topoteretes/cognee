from typing import List

from cognee.infrastructure.llm.extraction import extract_event_entities
from cognee.modules.engine.models import Event
from cognee.tasks.temporal_graph.models import EventWithEntities, EventEntityList


async def enrich_events(events: List[Event]) -> List[EventWithEntities]:
    """
    Enriches a list of events by extracting entities using an LLM.

    The function serializes event data into JSON, sends it to the LLM for
    entity extraction, and returns enriched events with associated entities.

    Args:
        events (List[Event]): A list of Event objects to be enriched.

    Returns:
        List[EventWithEntities]: A list of events augmented with extracted entities.
    """

    import json

    # Convert events to JSON format for LLM processing
    events_json = [
        {"event_name": event.name, "description": event.description or ""} for event in events
    ]

    events_json_str = json.dumps(events_json)

    # Extract entities from events
    entity_result = await extract_event_entities(events_json_str, EventEntityList)

    return entity_result.events
