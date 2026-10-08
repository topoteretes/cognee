"""Partial node updates preserve existing properties."""

import json
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_ladybug_update_node_merges_and_persists():
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter.__new__(LadybugAdapter)
    # get_node returns created_at/updated_at from inside the blob (not native columns);
    # the patch must carry them through, not strip them.
    adapter.get_node = AsyncMock(
        return_value={
            "id": "n1",
            "name": "Alice",
            "type": "Entity",
            "description": "keep me",
            "created_at": 1784623794966,
            "updated_at": 1784623794966,
        }
    )
    adapter.query = AsyncMock(return_value=[["n1"]])

    ok = await adapter.update_node("n1", {"text": "Prefer concise answers.", "turn_counter": 3})

    assert ok is True
    _, params = adapter.query.await_args.args
    stored = json.loads(params["properties"])
    assert stored["text"] == "Prefer concise answers."
    assert stored["turn_counter"] == 3
    assert stored["description"] == "keep me"  # unnamed field preserved
    assert stored["created_at"] == 1784623794966  # timestamps must survive the patch
    assert stored["updated_at"] == 1784623794966
    assert "id" not in stored and "name" not in stored  # core columns stay out of the blob
    assert params["id"] == "n1"


@pytest.mark.asyncio
async def test_ladybug_update_node_returns_false_for_missing_node():
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.get_node = AsyncMock(return_value=None)
    adapter.query = AsyncMock()

    assert (
        await adapter.update_node("missing", {"text": "Prefer concise answers.", "turn_counter": 3})
        is False
    )
    adapter.query.assert_not_awaited()
