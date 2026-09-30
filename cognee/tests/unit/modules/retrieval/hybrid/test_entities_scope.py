from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid.entities import build_entities


def _hit(entity_id, name):
    hit = MagicMock()
    hit.id = entity_id
    hit.payload = {"id": entity_id, "name": name}
    return hit


def _graph(nodes, edges):
    graph = MagicMock()
    graph.get_neighborhood = AsyncMock(return_value=(nodes, edges))
    return graph


ALICE = "alice-id"
ACME = "acme-id"
PERSON = "person-id"
WORKS_AT_TEXT = "Alice works at Acme."
NODES = [
    (ALICE, {"name": "Alice", "belongs_to_set": ["A"]}),
    (ACME, {"name": "Acme", "belongs_to_set": ["B"]}),
    (PERSON, {"name": "Person"}),
]
EDGES = [
    (ALICE, ACME, "works_at", {"edge_text": WORKS_AT_TEXT}),
    (ALICE, PERSON, "is_a", {}),
]


def _bullet_texts(entities):
    return [edge["text"] for edge in entities[0]["edges"]]


@pytest.mark.asyncio
async def test_scoped_build_drops_out_of_set_neighbors_and_keeps_type_edges():
    entities, reachable = await build_entities(
        _graph(NODES, EDGES),
        [_hit(ALICE, "Alice")],
        max_edges_per_entity=10,
        node_name=["A"],
    )

    assert WORKS_AT_TEXT not in _bullet_texts(entities)
    assert "Alice -- is_a -- Person" in _bullet_texts(entities)
    assert str(EdgeType.id_for(WORKS_AT_TEXT)) not in reachable


@pytest.mark.asyncio
async def test_unscoped_build_keeps_out_of_set_neighbors():
    entities, reachable = await build_entities(
        _graph(NODES, EDGES),
        [_hit(ALICE, "Alice")],
        max_edges_per_entity=10,
    )

    assert WORKS_AT_TEXT in _bullet_texts(entities)
    assert str(EdgeType.id_for(WORKS_AT_TEXT)) in reachable


@pytest.mark.asyncio
async def test_and_scope_requires_neighbor_to_carry_every_requested_set():
    nodes = [
        (ALICE, {"name": "Alice", "belongs_to_set": ["A", "B"]}),
        (ACME, {"name": "Acme", "belongs_to_set": ["A"]}),
    ]
    edges = [(ALICE, ACME, "works_at", {"edge_text": WORKS_AT_TEXT})]

    entities, _reachable = await build_entities(
        _graph(nodes, edges),
        [_hit(ALICE, "Alice")],
        max_edges_per_entity=10,
        node_name=["A", "B"],
        node_name_filter_operator="AND",
    )

    assert _bullet_texts(entities) == []
