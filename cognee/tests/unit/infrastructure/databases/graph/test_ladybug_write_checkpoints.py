"""Every Ladybug write is followed by a CHECKPOINT.

A write that is never checkpointed lives only in the WAL. If the process dies
before something else checkpoints, and the WAL cannot be replayed on the next
open, that write is gone from the graph.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


def _adapter(query_result=None):
    adapter = LadybugAdapter.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=query_result if query_result is not None else [])
    # add_edge is a batch of one, and the batch path serializes under this lock.
    adapter._bulk_write_lock = asyncio.Lock()
    return adapter


def _assert_checkpointed_after_write(adapter):
    queries = [call.args[0] for call in adapter.query.await_args_list]
    assert len(queries) >= 2, queries
    assert queries[-1] == "CHECKPOINT;"
    assert "CHECKPOINT" not in queries[0]


@pytest.mark.asyncio
async def test_add_node_checkpoints():
    adapter = _adapter()
    await adapter.add_node(SimpleNamespace(id="n1", name="Alice", type="Entity"))
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_add_edge_checkpoints():
    adapter = _adapter()
    await adapter.add_edge("n1", "n2", "knows", {})
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_delete_node_checkpoints():
    adapter = _adapter()
    await adapter.delete_node("n1")
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_delete_nodes_checkpoints():
    adapter = _adapter()
    await adapter.delete_nodes(["n1", "n2"])
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["remove_connection_to_predecessors_of", "remove_connection_to_successors_of"]
)
async def test_remove_connection_checkpoints(method):
    adapter = _adapter()
    await getattr(adapter, method)(["n1"], "knows")
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_update_node_checkpoints():
    adapter = _adapter(query_result=[["n1"]])
    adapter.get_node = AsyncMock(return_value={"id": "n1", "name": "Alice", "type": "Entity"})
    assert await adapter.update_node("n1", {"text": "hello"}) is True
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_update_node_without_match_does_not_checkpoint():
    adapter = _adapter(query_result=[])
    adapter.get_node = AsyncMock(return_value={"id": "n1", "name": "Alice", "type": "Entity"})
    assert await adapter.update_node("n1", {"text": "hello"}) is False
    queries = [call.args[0] for call in adapter.query.await_args_list]
    assert "CHECKPOINT;" not in queries


@pytest.mark.asyncio
async def test_set_node_feedback_weights_checkpoints():
    adapter = _adapter(query_result=[["n1"]])
    adapter.get_nodes = AsyncMock(return_value=[{"id": "n1", "type": "Entity"}])
    assert await adapter.set_node_feedback_weights({"n1": 0.9}) == {"n1": True}
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_set_node_truth_state_checkpoints():
    adapter = _adapter(query_result=[["n1"]])
    adapter.get_nodes = AsyncMock(return_value=[{"id": "n1", "type": "Entity"}])
    result = await adapter.set_node_truth_state({"n1": {"truth_alignment": [1.0]}})
    assert result == {"n1": True}
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_set_edge_feedback_weights_checkpoints():
    adapter = _adapter(query_result=[["e1"]])
    adapter._fetch_edge_rows_by_object_ids = AsyncMock(
        return_value=[
            {
                "from_id": "n1",
                "to_id": "n2",
                "relationship_name": "knows",
                "properties": '{"edge_object_id": "e1"}',
                "edge_object_id_json": None,
            }
        ]
    )
    assert await adapter.set_edge_feedback_weights({"e1": 0.9}) == {"e1": True}
    _assert_checkpointed_after_write(adapter)


@pytest.mark.asyncio
async def test_set_feedback_weights_without_updates_does_not_write():
    adapter = _adapter()
    adapter.get_nodes = AsyncMock(return_value=[])
    assert await adapter.set_node_feedback_weights({"n1": 0.9}) == {"n1": False}
    adapter.query.assert_not_awaited()
