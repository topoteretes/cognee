"""Bulk node/edge writes must be chunked so no single statement can run past
the subprocess engine's per-call deadline on large graphs (COG: ladybug
ingestion of e.g. 30k-fact code graphs previously sent one statement for all
rows and could never finish)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import (
    LadybugAdapter,
    _WRITE_CHUNK_SIZE,
)


def _adapter_with_mocked_writes():
    adapter = object.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=[])
    adapter.checkpoint = AsyncMock()
    return adapter


def _fake_nodes(count):
    return [
        SimpleNamespace(id=f"node-{index}", name=f"n{index}", type="Node") for index in range(count)
    ]


def _fake_edges(count):
    return [(f"from-{index}", f"to-{index}", "relates_to", {}) for index in range(count)]


@pytest.mark.asyncio
async def test_add_nodes_chunks_large_batches():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE * 2 + 1

    await adapter.add_nodes(_fake_nodes(total))

    assert adapter.query.await_count == 3
    chunk_sizes = [len(call.args[1]["nodes"]) for call in adapter.query.await_args_list]
    assert chunk_sizes == [_WRITE_CHUNK_SIZE, _WRITE_CHUNK_SIZE, 1]
    adapter.checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_nodes_small_batch_is_single_statement():
    adapter = _adapter_with_mocked_writes()

    await adapter.add_nodes(_fake_nodes(5))

    assert adapter.query.await_count == 1


@pytest.mark.asyncio
async def test_add_edges_chunks_large_batches():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE + 1

    await adapter.add_edges(_fake_edges(total))

    assert adapter.query.await_count == 2
    chunk_sizes = [len(call.args[1]["edges"]) for call in adapter.query.await_args_list]
    assert chunk_sizes == [_WRITE_CHUNK_SIZE, 1]
    adapter.checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_edges_matches_endpoints_by_primary_key():
    """The edge MERGE must seek endpoints via property-map matches (index
    lookups), not a cartesian MATCH + WHERE that plans as a scan."""
    adapter = _adapter_with_mocked_writes()

    await adapter.add_edges(_fake_edges(1))

    query = adapter.query.await_args_list[0].args[0]
    assert "MATCH (from:Node {id: edge.from_id})" in query
    assert "MATCH (to:Node {id: edge.to_id})" in query
    assert "MATCH (from:Node), (to:Node)" not in query
