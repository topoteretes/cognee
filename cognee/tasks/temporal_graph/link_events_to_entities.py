"""Deterministic event-to-entity linking.

After the standard extraction has put ``Entity`` nodes on a chunk and the temporal
extraction has put ``Event`` nodes next to them, this step connects each event to
the entities of the *same chunk* whose name occurs in the event's text. No LLM
call, no new nodes: the events point at entities the graph already has, so the
timeline and the entity graph share vertices instead of running in parallel.
"""

import re
from typing import Iterable, List

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.engine.models import Entity, Event

INVOLVES = "involves"
# Below this length a name is too generic to match reliably ("a", "it", "us").
MIN_NAME_LENGTH = 3


def _entities_in(contains: Iterable) -> List[Entity]:
    """Entities on a chunk arrive bare or as ``(Edge, Entity)`` tuples; yield the entities."""
    entities: List[Entity] = []
    for item in contains or []:
        candidate = item[1] if isinstance(item, tuple) and len(item) == 2 else item
        if isinstance(candidate, Entity):
            entities.append(candidate)
    return entities


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


def link_event_to_entities(event: Event, entities: Iterable[Entity]) -> int:
    """Attach every entity mentioned in the event's name or description. Returns the count."""
    text = f"{event.name or ''}\n{event.description or ''}".lower()
    already = {
        str(target.id)
        for _edge, targets in (event.attributes or [])
        for target in (targets if isinstance(targets, list) else [targets])
        if hasattr(target, "id")
    }

    linked = 0
    for entity in entities:
        name = (entity.name or "").strip().lower()
        if len(name) < MIN_NAME_LENGTH or str(entity.id) in already:
            continue
        if _mentions(text, name):
            if event.attributes is None:
                event.attributes = []
            event.attributes.append((Edge(relationship_type=INVOLVES), [entity]))
            already.add(str(entity.id))
            linked += 1
    return linked


def link_events_to_entities(data_chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Link the events of each chunk to the entities extracted from that chunk."""
    for chunk in data_chunks:
        contains = chunk.contains or []
        entities = _entities_in(contains)
        if not entities:
            continue
        for item in contains:
            if isinstance(item, Event):
                link_event_to_entities(item, entities)
    return data_chunks
