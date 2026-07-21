from unittest.mock import AsyncMock

import pytest

from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid.entities import build_entities


def _hit(entity_id: str, sets: list[str]):
    return {
        "id": entity_id,
        "name": entity_id,
        "belongs_to_set": sets,
    }


@pytest.mark.asyncio
async def test_scoped_neighborhood_drops_out_of_scope_endpoint_and_edge():
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [
            ("seed", {"name": "Seed", "belongs_to_set": ["KEEP"]}),
            ("kept", {"name": "Kept", "belongs_to_set": ["KEEP"]}),
            ("other", {"name": "Other", "belongs_to_set": ["OTHER"]}),
        ],
        [
            ("seed", "kept", "links", {"edge_text": "Seed links Kept"}),
            ("seed", "other", "links", {"edge_text": "Seed links Other"}),
            (
                "seed",
                "kept",
                "private_link",
                {"edge_text": "Private edge", "belongs_to_set": ["OTHER"]},
            ),
        ],
    )

    entities = await build_entities(
        graph,
        [_hit("seed", ["KEEP"])],
        10,
        node_name=["KEEP"],
        node_name_filter_operator="AND",
    )

    assert [edge["text"] for edge in entities[0]["edges"]] == ["Seed links Kept"]


@pytest.mark.asyncio
async def test_scoped_entity_seed_without_requested_membership_is_rejected():
    graph = AsyncMock()

    entities = await build_entities(
        graph,
        [_hit("seed", ["OTHER"])],
        10,
        node_name=["KEEP"],
    )

    assert entities == []
    graph.get_neighborhood.assert_not_awaited()


@pytest.mark.asyncio
async def test_scoped_entity_accepts_serialized_nodeset_objects():
    graph = AsyncMock()
    graph.get_neighborhood.return_value = ([], [])

    entities = await build_entities(
        graph,
        [_hit("seed", [{"id": "set-id", "name": "KEEP"}])],
        10,
        node_name=["KEEP"],
    )

    assert [entity["id"] for entity in entities] == ["seed"]


@pytest.mark.asyncio
async def test_query_ranked_edge_beats_unranked_type_edge_before_cap():
    relevant = "Lisbon office owns HarborLens"
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [
            ("seed", {"name": "Lisbon office"}),
            ("project", {"name": "HarborLens"}),
            ("type", {"name": "Office"}),
        ],
        [
            ("seed", "type", "is_a", {"edge_text": "Lisbon office is a Office"}),
            ("seed", "project", "owns", {"edge_text": relevant}),
        ],
    )

    entities = await build_entities(
        graph,
        [_hit("seed", [])],
        1,
        edge_ranks={str(EdgeType.id_for(relevant)): 0},
    )

    assert [edge["text"] for edge in entities[0]["edges"]] == [relevant]


@pytest.mark.asyncio
async def test_query_ranked_nonrendered_edge_is_reserved_as_fact_evidence():
    first = "Alice works at Acme"
    second = "Alice founded Initech"
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [
            ("seed", {"name": "Alice"}),
            ("acme", {"name": "Acme"}),
            ("initech", {"name": "Initech"}),
        ],
        [
            ("seed", "acme", "works_at", {"edge_text": first}),
            ("seed", "initech", "founded", {"edge_text": second}),
        ],
    )

    entities = await build_entities(
        graph,
        [_hit("seed", [])],
        1,
        edge_ranks={
            str(EdgeType.id_for(first)): 0,
            str(EdgeType.id_for(second)): 1,
        },
    )

    assert [edge["text"] for edge in entities[0]["edges"]] == [first]
    assert [edge["text"] for edge in entities[0]["fact_evidence"]] == [second]


@pytest.mark.asyncio
async def test_edge_preserves_source_and_temporal_provenance():
    graph = AsyncMock()
    graph.get_neighborhood.return_value = (
        [("seed", {"name": "Alice"}), ("target", {"name": "Acme"})],
        [
            (
                "seed",
                "target",
                "works_at",
                {
                    "edge_text": "Alice works at Acme",
                    "source_chunk_id": "chunk-1",
                    "valid_from": "2026-03-01",
                    "untrusted_internal_value": "not exposed",
                },
            )
        ],
    )

    entities = await build_entities(graph, [_hit("seed", [])], 10)

    assert entities[0]["edges"][0]["provenance"] == {
        "source_chunk_id": "chunk-1",
        "valid_from": "2026-03-01",
    }
