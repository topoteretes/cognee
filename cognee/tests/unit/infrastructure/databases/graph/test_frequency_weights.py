import json
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


@pytest.mark.asyncio
async def test_ladybug_get_node_frequency_weights_defaults_to_zero():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.get_nodes = AsyncMock(
        return_value=[
            {"id": "n1", "frequency_weight": "2.5"},
            {"id": "n2"},
            {"id": "n3", "frequency_weight": "bad"},
        ]
    )

    weights = await adapter.get_node_frequency_weights(["n1", "n2", "n3", "missing"])

    assert weights == {"n1": 2.5, "n2": 0.0, "n3": 0.0}


@pytest.mark.asyncio
async def test_ladybug_set_node_frequency_weights_preserves_properties():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.get_nodes = AsyncMock(
        return_value=[
            {
                "id": "n1",
                "name": "Node 1",
                "type": "Entity",
                "description": "kept",
                "feedback_weight": 0.7,
            }
        ]
    )
    adapter._execute_node_frequency_updates = AsyncMock(return_value={"n1"})

    result = await adapter.set_node_frequency_weights({"n1": 3.0, "missing": 1.0})

    assert result == {"n1": True, "missing": False}
    updates = adapter._execute_node_frequency_updates.await_args.args[0]
    properties = json.loads(updates[0]["properties"])
    assert properties["description"] == "kept"
    assert properties["feedback_weight"] == 0.7
    assert properties["frequency_weight"] == 3.0
    assert "id" not in properties
    assert "type" not in properties


@pytest.mark.asyncio
async def test_ladybug_get_edge_frequency_weights_defaults_to_zero():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter._fetch_edge_rows_by_object_ids = AsyncMock(
        return_value=[
            {
                "edge_object_id_json": json.dumps("e1"),
                "properties": json.dumps({"edge_object_id": "e1", "frequency_weight": "4"}),
            },
            {
                "edge_object_id_json": json.dumps("e2"),
                "properties": json.dumps({"edge_object_id": "e2"}),
            },
            {
                "edge_object_id_json": json.dumps("e3"),
                "properties": json.dumps({"edge_object_id": "e3", "frequency_weight": "bad"}),
            },
        ]
    )

    weights = await adapter.get_edge_frequency_weights(["e1", "e2", "e3", "missing"])

    assert weights == {"e1": 4.0, "e2": 0.0, "e3": 0.0}


@pytest.mark.asyncio
async def test_ladybug_set_edge_frequency_weights_preserves_properties():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter._fetch_edge_rows_by_object_ids = AsyncMock(
        return_value=[
            {
                "from_id": "n1",
                "to_id": "n2",
                "relationship_name": "REL",
                "edge_object_id_json": json.dumps("e1"),
                "properties": json.dumps(
                    {"edge_object_id": "e1", "edge_text": "kept", "feedback_weight": 0.6}
                ),
            }
        ]
    )
    adapter._execute_edge_frequency_updates = AsyncMock(return_value={"e1"})

    result = await adapter.set_edge_frequency_weights({"e1": 2.0, "missing": 1.0})

    assert result == {"e1": True, "missing": False}
    updates = adapter._execute_edge_frequency_updates.await_args.args[0]
    properties = json.loads(updates[0]["properties"])
    assert properties["edge_object_id"] == "e1"
    assert properties["edge_text"] == "kept"
    assert properties["feedback_weight"] == 0.6
    assert properties["frequency_weight"] == 2.0


@pytest.mark.asyncio
async def test_neo4j_get_node_frequency_weights_uses_zero_default():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(
        return_value=[
            {"node_id": "n1", "frequency_weight": 5},
            {"node_id": "n2", "frequency_weight": 0.0},
        ]
    )

    weights = await adapter.get_node_frequency_weights(["n1", "n2", "", None])

    assert weights == {"n1": 5.0, "n2": 0.0}
    assert adapter.query.await_args.args[1] == {
        "node_ids": ["n1", "n2"],
        "default_weight": 0.0,
    }


@pytest.mark.asyncio
async def test_neo4j_set_node_frequency_weights_returns_per_id_status():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter._execute_node_frequency_updates = AsyncMock(return_value={"n1"})

    result = await adapter.set_node_frequency_weights({"n1": 2.0, "missing": 1.0, "": 3.0})

    assert result == {"n1": True, "missing": False, "": False}
    items = adapter._execute_node_frequency_updates.await_args.args[0]
    assert items == [
        {"node_id": "n1", "frequency_weight": 2.0},
        {"node_id": "missing", "frequency_weight": 1.0},
    ]


@pytest.mark.asyncio
async def test_neo4j_get_edge_frequency_weights_uses_zero_default():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter.query = AsyncMock(
        return_value=[
            {"edge_object_id": "e1", "frequency_weight": 6},
            {"edge_object_id": "e2", "frequency_weight": 0.0},
        ]
    )

    weights = await adapter.get_edge_frequency_weights(["e1", "e2", "", None])

    assert weights == {"e1": 6.0, "e2": 0.0}
    assert adapter.query.await_args.args[1] == {
        "edge_object_ids": ["e1", "e2"],
        "default_weight": 0.0,
    }


@pytest.mark.asyncio
async def test_neo4j_set_edge_frequency_weights_returns_per_id_status():
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    adapter._execute_edge_frequency_updates = AsyncMock(return_value={"e1"})

    result = await adapter.set_edge_frequency_weights({"e1": 2.0, "missing": 1.0, "": 3.0})

    assert result == {"e1": True, "missing": False, "": False}
    items = adapter._execute_edge_frequency_updates.await_args.args[0]
    assert items == [
        {"edge_object_id": "e1", "frequency_weight": 2.0},
        {"edge_object_id": "missing", "frequency_weight": 1.0},
    ]
