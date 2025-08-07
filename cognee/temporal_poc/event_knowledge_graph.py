from typing import List, Type
from pydantic import BaseModel

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.utils import generate_node_id, generate_node_name
from cognee.temporal_poc.models.models import EventEntityList
from cognee.temporal_poc.datapoints.datapoints import Event
from cognee.temporal_poc.models.models import EventWithEntities

ENTITY_EXTRACTION_SYSTEM_PROMPT = """For the purposes of building event-based knowledge graphs, you are tasked with extracting highly granular entities from events text. An entity is any distinct, identifiable thing, person, place, object, organization, concept, or phenomenon that can be named, referenced, or described in the event context. This includes but is not limited to: people, places, objects, organizations, concepts, events, processes, states, conditions, properties, attributes, roles, functions, and any other meaningful referents that contribute to understanding the event.
**Temporal Entity Exclusion**: Do not extract timestamp-like entities (dates, times, durations) as these are handled separately. However, extract named temporal periods, eras, historical epochs, and culturally significant time references
## Input Format
The input will be a list of dictionaries, each containing:
- `event_name`: The name of the event
- `description`: The description of the event

## Task
For each event, extract all entities mentioned in the event description and determine their relationship to the event.

## Output Format
Return the same enriched JSON with an additional key in each dictionary: `attributes`.

The `attributes` should be a list of dictionaries, each containing:
- `entity`: The name of the entity
- `entity_type`: The type/category of the entity (person, place, organization, object, concept, etc.)
- `relationship`: A concise description of how the entity relates to the event

## Requirements
- **Be extremely thorough** - extract EVERY non-temporal entity mentioned, no matter how small, obvious, or seemingly insignificant
- **After you are done with obvious entities, every noun, pronoun, proper noun, and named reference =  one entity**
- We expect rich entity networks from any event, easily reaching a dozens of entities per event
- Granularity and richness of the entity extraction is key to our success and is of utmost importance
- **Do not skip any entities** - if you're unsure whether something is an entity, extract it anyway
- Use the event name for context when determining relationships
- Relationships should be technical with one or at most two words. If two words, use underscore camelcase style
- Relationships could imply general meaning like: subject, object, participant, recipient, agent, instrument, tool, source, cause, effect, purpose, manner, resource, etc.
- You can combine two words to form a relationship name: subject_role, previous_owner, etc.
- Focus on how the entity specifically relates to the event
"""


async def extract_event_entities(
    content: str, response_model: Type[BaseModel], system_prompt: str = None
):
    """Extract event entities from content using LLM."""

    if system_prompt is None:
        system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT

    content_graph = await LLMGateway.acreate_structured_output(
        content, system_prompt, response_model
    )

    return content_graph


async def enrich_events(events: List[Event]) -> List[EventWithEntities]:
    """Extract entities from events and return enriched events."""
    import json

    # Convert events to JSON format for LLM processing
    events_json = [
        {"event_name": event.name, "description": event.description or ""} for event in events
    ]

    events_json_str = json.dumps(events_json)

    # Extract entities from events
    entity_result = await extract_event_entities(events_json_str, EventEntityList)

    return entity_result.events


def add_entities_to_event(event: Event, event_with_entities: EventWithEntities) -> None:
    """Add entities to event via attributes field."""
    if not event_with_entities.attributes:
        return

    # Create entity types cache
    entity_types = {}

    # Process each attribute
    for attribute in event_with_entities.attributes:
        # Get or create entity type
        entity_type = get_or_create_entity_type(entity_types, attribute.entity_type)

        # Create entity
        entity_id = generate_node_id(attribute.entity)
        entity_name = generate_node_name(attribute.entity)
        entity = Entity(
            id=entity_id,
            name=entity_name,
            is_a=entity_type,
            description=f"Entity {attribute.entity} of type {attribute.entity_type}",
            ontology_valid=False,
            belongs_to_set=None,
        )

        # Create edge
        edge = Edge(relationship_type=attribute.relationship)

        # Add to event attributes
        if event.attributes is None:
            event.attributes = []
        event.attributes.append((edge, [entity]))


def get_or_create_entity_type(entity_types: dict, entity_type_name: str) -> EntityType:
    """Get existing entity type or create new one."""
    if entity_type_name not in entity_types:
        type_id = generate_node_id(entity_type_name)
        type_name = generate_node_name(entity_type_name)
        entity_type = EntityType(
            id=type_id,
            name=type_name,
            type=type_name,
            description=f"Type for {entity_type_name}",
            ontology_valid=False,
        )
        entity_types[entity_type_name] = entity_type

    return entity_types[entity_type_name]


async def extract_event_knowledge_graph(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Extract events from chunks and enrich them with entities."""
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


async def process_event_knowledge_graph(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Process document chunks for event knowledge graph construction."""
    return await extract_event_knowledge_graph(data_chunks)
