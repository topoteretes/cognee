import json
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


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
