import json
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


@pytest.fixture(autouse=True)
def _relational_db_for_unit_tests():
    pass


def test_build_node_truth_state_updates_preserves_properties_and_adds_epoch():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    updates = adapter._build_node_truth_state_updates(
        [{"id": "n1", "type": "DocumentChunk", "text": "hello", "feedback_weight": 0.8}],
        {"n1": {"truth_alignment": [1.0, 0.0], "truth_epoch": 7}},
    )

    assert len(updates) == 1
    properties = json.loads(updates[0]["properties"])
    assert properties["text"] == "hello"
    assert properties["feedback_weight"] == 0.8
    assert properties["truth_alignment"] == [1.0, 0.0]
    assert properties["truth_epoch"] == 7
    assert "id" not in properties
    assert "type" not in properties


@pytest.mark.asyncio
async def test_get_node_truth_state_returns_alignment_and_epoch():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.get_nodes = AsyncMock(
        return_value=[
            {"id": "n1", "truth_alignment": [0.1, 0.2], "truth_epoch": "3"},
            {"id": "n2", "truth_alignment": "bad", "truth_epoch": "bad"},
        ]
    )

    state = await adapter.get_node_truth_state(["n1", "n2"])

    assert state["n1"] == {"truth_alignment": [0.1, 0.2], "truth_epoch": 3}
    assert state["n2"] == {"truth_alignment": [], "truth_epoch": None}


@pytest.mark.asyncio
async def test_get_nodeset_subgraph_and_ignores_missing_names():
    adapter = LadybugAdapter.__new__(LadybugAdapter)

    adapter.query = AsyncMock(
        side_effect=[
            [("primary-a",)],  # primary nodes: "Alpha" matched
            [("shared",)],  # neighbor nodes
            [
                ("primary-a", "Alpha", "Entity", {}),
                ("shared", "Shared", "Other", {}),
            ],  # nodes
            [
                ("primary-a", "shared", "R", {}),
                ("shared", "primary-a", "R", {}),
            ],  # edges
        ]
    )

    class Entity:
        pass

    nodes, edges = await adapter.get_nodeset_subgraph(
        Entity, ["Alpha", "Missing"], node_name_filter_operator="AND"
    )

    _, neighbor_params = adapter.query.await_args_list[1].args
    assert neighbor_params["primary_count"] == 1

    assert {node_id for node_id, _ in nodes} == {"primary-a", "shared"}
    assert len(edges) == 2


@pytest.mark.asyncio
async def test_get_nodeset_subgraph_rejects_invalid_operator():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[("primary-a",)])

    class Entity:
        pass

    with pytest.raises(ValueError, match="node_name_filter_operator must be 'OR' or 'AND'"):
        await adapter.get_nodeset_subgraph(Entity, ["Alpha"], node_name_filter_operator="INVALID")

    assert adapter.query.await_count == 1


@pytest.mark.asyncio
async def test_get_nodeset_subgraph_returns_each_edge_once():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(
        side_effect=[
            [("primary-a",)],
            [("shared",)],
            [
                ("primary-a", "Alpha", "Entity", {}),
                ("shared", "Shared", "Other", {}),
            ],
            [
                ("primary-a", "shared", "R", {}),
            ],
        ]
    )

    class Entity:
        pass

    nodes, edges = await adapter.get_nodeset_subgraph(
        Entity, ["Alpha"], node_name_filter_operator="OR"
    )

    assert {node_id for node_id, _ in nodes} == {"primary-a", "shared"}
    assert len(edges) == 1
    assert edges == [("primary-a", "shared", "R", {})]

    edge_query, _ = adapter.query.await_args_list[3].args
    assert "a.id < b.id" in edge_query
