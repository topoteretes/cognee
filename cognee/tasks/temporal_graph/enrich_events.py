from typing import List

from cognee.infrastructure.llm import LLMGateway
from cognee.modules.engine.models import Event
from cognee.tasks.temporal_graph.models import EventWithEntities,EventEntityList

async def enrich_events(events: List[Event]) -> List[EventWithEntities]:
    """Extract entities from events and return enriched events."""
    import json

    # Convert events to JSON format for LLM processing
    events_json = [
        {"event_name": event.name, "description": event.description or ""} for event in events
    ]

    events_json_str = json.dumps(events_json)

    # Extract entities from events
    entity_result = await LLMGateway.extract_event_entities(events_json_str, EventEntityList)

    return entity_result.events