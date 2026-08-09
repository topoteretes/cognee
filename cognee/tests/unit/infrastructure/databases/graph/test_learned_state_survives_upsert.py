"""Learned graph state (feedback_weight, truth coords) must survive re-cognify.

Node and edge upserts replace the stored JSON property blob (Ladybug) or merge
incoming defaults over stored values (Neo4j `+=`). These tests pin the Ladybug
preservation path: the adapter reads existing learned fields before writing and
carries them into the new blob, so re-ingesting the same document cannot reset
what the feedback loop and truth builds learned.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.modules.graph.utils.prepare_edges_for_storage import (
    ensure_default_edge_properties,
)


def _adapter():
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[])
    adapter.checkpoint = AsyncMock()
    return adapter


def _incoming_node(node_id="n1"):
    # vars()-compatible stand-in for a DataPoint carrying the model default weight.
    return SimpleNamespace(
        id=node_id,
        name="Entity A",
        type="Entity",
        text="hello",
        feedback_weight=0.5,
    )


@pytest.mark.asyncio
async def test_add_nodes_preserves_learned_fields_on_existing_node():
    adapter = _adapter()
    adapter.get_nodes = AsyncMock(
        return_value=[
            {
                "id": "n1",
                "type": "Entity",
                "feedback_weight": 0.9,
                "truth_alignment": [1.0, 0.0],
                "truth_epoch": 2,
            }
        ]
    )

    await adapter.add_nodes([_incoming_node("n1")])

    (_query, params), _ = adapter.query.await_args
    written = json.loads(params["nodes"][0]["properties"])
    assert written["feedback_weight"] == 0.9
    assert written["truth_alignment"] == [1.0, 0.0]
    assert written["truth_epoch"] == 2
    assert written["text"] == "hello"


@pytest.mark.asyncio
async def test_add_nodes_fresh_node_keeps_incoming_default():
    adapter = _adapter()
    adapter.get_nodes = AsyncMock(return_value=[])

    await adapter.add_nodes([_incoming_node("n-new")])

    (_query, params), _ = adapter.query.await_args
    written = json.loads(params["nodes"][0]["properties"])
    assert written["feedback_weight"] == 0.5
    assert "truth_alignment" not in written


@pytest.mark.asyncio
async def test_add_edges_preserves_learned_feedback_weight():
    adapter = _adapter()
    adapter._fetch_edge_rows_by_object_ids = AsyncMock(
        return_value=[
            {
                "from_id": "a",
                "to_id": "b",
                "relationship_name": "rel",
                "edge_object_id_json": json.dumps("e1"),
                "properties": json.dumps({"edge_object_id": "e1", "feedback_weight": 0.83}),
            }
        ]
    )

    await adapter.add_edges([("a", "b", "rel", {"edge_object_id": "e1", "weight": 1.0})])

    (_query, params), _ = adapter.query.await_args
    written = json.loads(params["edges"][0]["properties"])
    assert written["feedback_weight"] == 0.83
    assert written["weight"] == 1.0


@pytest.mark.asyncio
async def test_add_edges_fresh_edge_carries_no_feedback_weight():
    adapter = _adapter()
    adapter._fetch_edge_rows_by_object_ids = AsyncMock(return_value=[])

    await adapter.add_edges([("a", "b", "rel", {"edge_object_id": "e-new"})])

    (_query, params), _ = adapter.query.await_args
    written = json.loads(params["edges"][0]["properties"])
    assert "feedback_weight" not in written


def test_prepare_edges_no_longer_injects_default_feedback_weight():
    """Consumers default to 0.5 at read time; injecting it at prepare time made
    every re-cognify overwrite learned edge weights."""
    (edge,) = ensure_default_edge_properties([("a", "b", "rel", {})])
    _source, _target, _rel, props = edge
    assert "feedback_weight" not in props
    assert props["edge_object_id"]

    (explicit_edge,) = ensure_default_edge_properties([("a", "b", "rel", {"feedback_weight": 0.9})])
    assert explicit_edge[3]["feedback_weight"] == 0.9
