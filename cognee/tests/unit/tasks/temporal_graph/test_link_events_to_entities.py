"""Deterministic event-to-entity linking (SDK-80).

With temporal_cognify the events of a chunk must point at the entities the standard
extraction produced for that chunk — no LLM call, no new nodes — so the timeline and
the knowledge graph share vertices.
"""

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import Entity, EntityType, Event
from cognee.tasks.temporal_graph.link_events_to_entities import (
    INVOLVES,
    link_event_to_entities,
    link_events_to_entities,
)


def _entity(name: str) -> Entity:
    return Entity(
        name=name,
        description=f"{name} entity",
        is_a=EntityType(name="person", type="person", description="p"),
    )


def _chunk(contains):
    class _Chunk:
        pass

    chunk = _Chunk()
    chunk.contains = contains
    return chunk


def _linked_ids(event: Event):
    return [str(targets[0].id) for _edge, targets in (event.attributes or [])]


def test_links_entities_mentioned_in_name_or_description():
    attaphol = _entity("attaphol buspakom")
    buriram = _entity("buriram united")
    other = _entity("arnulf øverland")
    event = Event(
        name="Attaphol joins Buriram United",
        description="In 2010 Attaphol Buspakom moved to Buriram United F.C.",
    )

    linked = link_event_to_entities(event, [attaphol, buriram, other])

    assert linked == 2
    assert _linked_ids(event) == [str(attaphol.id), str(buriram.id)]
    edge, targets = event.attributes[0]
    assert isinstance(edge, Edge) and edge.relationship_type == INVOLVES
    assert targets == [attaphol]


def test_matching_is_word_bounded_and_case_insensitive():
    art = _entity("Art")
    event = Event(name="Party", description="A PARTY for the ART department")

    assert link_event_to_entities(event, [art]) == 1  # "ART" matches, "pARTy" does not

    event2 = Event(name="Party", description="a party")
    assert link_event_to_entities(event2, [art]) == 0
    assert event2.attributes is None


def test_short_names_are_skipped():
    event = Event(name="Meeting", description="he met us at 5")
    assert link_event_to_entities(event, [_entity("he"), _entity("us"), _entity("5")]) == 0


def test_does_not_link_the_same_entity_twice():
    oslo = _entity("oslo")
    event = Event(name="Move to Oslo", description="Moved to Oslo in 1946.")

    assert link_event_to_entities(event, [oslo]) == 1
    assert link_event_to_entities(event, [oslo]) == 0
    assert len(event.attributes) == 1


def test_links_only_events_to_entities_of_the_same_chunk():
    einstein = _entity("einstein")
    bohr = _entity("bohr")
    event_a = Event(name="Einstein publishes", description="Einstein publishes in 1905")
    event_b = Event(name="Bohr publishes", description="Bohr publishes in 1913")
    # Entities may arrive bare or wrapped as (Edge, Entity) tuples.
    chunk_a = _chunk([(Edge(relationship_type="contains"), einstein), event_a])
    chunk_b = _chunk([bohr, event_b])
    chunk_without_entities = _chunk([Event(name="Bohr again", description="Bohr")])

    link_events_to_entities([chunk_a, chunk_b, chunk_without_entities])

    assert _linked_ids(event_a) == [str(einstein.id)]
    assert _linked_ids(event_b) == [str(bohr.id)]
    assert chunk_without_entities.contains[0].attributes is None


def test_handles_chunks_without_contains():
    assert link_events_to_entities([_chunk(None), _chunk([])]) is not None
