"""Unit tests for bi-temporal node validity (issue #3700 / SDK-200).

Covers the ``valid_to`` field, the ``is_valid`` staleness check, and — crucially —
that ``close_node`` actually calls a real graph update and returns its result (the
original PR swallowed an ``AttributeError`` and was a silent no-op on every backend),
plus that ``LadybugAdapter.update_node`` merges a patch without disturbing other
fields.
"""

import json
import time
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.tasks.storage.close_node import close_node, is_valid

# close_node imports get_graph_engine lazily from this module, so that is the patch target.
CLOSE_NODE_ENGINE = "cognee.infrastructure.databases.graph.get_graph_engine"


# ---------------------------------------------------------------------------
# DataPoint.valid_to field
# ---------------------------------------------------------------------------


def test_valid_to_defaults_to_none():
    assert DataPoint(id=uuid4()).valid_to is None


def test_valid_to_can_be_set():
    ts = int(time.time() * 1000)
    assert DataPoint(id=uuid4(), valid_to=ts).valid_to == ts


def test_valid_to_survives_json_round_trip():
    ts = int(time.time() * 1000)
    dp = DataPoint(id=uuid4(), valid_to=ts)
    assert DataPoint.from_json(dp.to_json()).valid_to == ts


# ---------------------------------------------------------------------------
# is_valid()
# ---------------------------------------------------------------------------


def test_is_valid_when_valid_to_is_none():
    assert is_valid(DataPoint(id=uuid4()))


def test_is_not_valid_when_valid_to_in_past():
    dp = DataPoint(id=uuid4(), valid_to=int(time.time() * 1000) - 60_000)
    assert not is_valid(dp)


def test_is_valid_when_valid_to_in_future():
    dp = DataPoint(id=uuid4(), valid_to=int(time.time() * 1000) + 3_600_000)
    assert is_valid(dp)


def test_is_valid_with_dict_node():
    assert is_valid({"valid_to": None})
    assert not is_valid({"valid_to": int(time.time() * 1000) - 1000})


def test_is_valid_at_explicit_timestamp():
    ts = 1_000_000_000_000
    dp = DataPoint(id=uuid4(), valid_to=ts + 1)
    assert is_valid(dp, at_ms=ts)
    assert not is_valid(dp, at_ms=ts + 2)


# ---------------------------------------------------------------------------
# close_node() wiring — it must call a real update and surface the result
# (the regression the original silent no-op could not catch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_node_persists_via_update_node():
    engine = AsyncMock()
    engine.update_node = AsyncMock(return_value=True)
    node_id = uuid4()
    with patch(CLOSE_NODE_ENGINE, AsyncMock(return_value=engine)):
        result = await close_node(node_id, at_ms=123456)
    assert result is True
    engine.update_node.assert_awaited_once_with(str(node_id), {"valid_to": 123456})


@pytest.mark.asyncio
async def test_close_node_returns_false_when_node_missing():
    engine = AsyncMock()
    engine.update_node = AsyncMock(return_value=False)
    with patch(CLOSE_NODE_ENGINE, AsyncMock(return_value=engine)):
        assert await close_node(uuid4()) is False


@pytest.mark.asyncio
async def test_close_node_handles_backend_without_update_node():
    """An unsupported backend must warn and return False, never raise or silently succeed."""
    engine = AsyncMock()
    engine.update_node = AsyncMock(side_effect=NotImplementedError)
    with patch(CLOSE_NODE_ENGINE, AsyncMock(return_value=engine)):
        assert await close_node(uuid4()) is False


@pytest.mark.asyncio
async def test_close_node_is_last_write_wins_on_reclose():
    """Re-closing overwrites valid_to with no guard, even moving the point earlier."""
    engine = AsyncMock()
    engine.update_node = AsyncMock(return_value=True)
    node_id = uuid4()
    with patch(CLOSE_NODE_ENGINE, AsyncMock(return_value=engine)):
        await close_node(node_id, at_ms=2000)
        await close_node(node_id, at_ms=1000)  # earlier timestamp is still applied
    assert engine.update_node.await_args_list[0].args == (str(node_id), {"valid_to": 2000})
    assert engine.update_node.await_args_list[1].args == (str(node_id), {"valid_to": 1000})


# ---------------------------------------------------------------------------
# LadybugAdapter.update_node — read-modify-write merges the patch, keeps the rest
# ---------------------------------------------------------------------------


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

    ok = await adapter.update_node("n1", {"valid_to": 999})

    assert ok is True
    _, params = adapter.query.await_args.args
    stored = json.loads(params["properties"])
    assert stored["valid_to"] == 999
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

    assert await adapter.update_node("missing", {"valid_to": 1}) is False
    adapter.query.assert_not_awaited()
