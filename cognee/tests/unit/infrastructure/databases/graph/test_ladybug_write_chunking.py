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
    assert "MATCH (from:Node {id: edge.from_id}), (to:Node {id: edge.to_id})" in query
    assert "MATCH (from:Node), (to:Node)" not in query


@pytest.mark.asyncio
async def test_add_edges_matches_both_endpoints_in_one_clause():
    """Both endpoints must be bound by a single comma-separated MATCH.

    Two separate MATCH clauses hit the row-driven primary-key lookup added in
    ladybug 0.19.0 (LadybugDB/ladybug#722) and segfault the engine mid-write
    (COG-6185). The comma form is equally fast and writes an identical graph,
    so this is the only shape allowed here.
    """
    adapter = _adapter_with_mocked_writes()

    await adapter.add_edges(_fake_edges(1))

    query = adapter.query.await_args_list[0].args[0]
    match_clauses = [line for line in query.splitlines() if line.strip().startswith("MATCH ")]
    assert len(match_clauses) == 1, f"endpoints must bind in one MATCH clause, got: {match_clauses}"
    assert "MATCH (to:Node {id: edge.to_id})" not in query


def _fake_edge_identities(count):
    from cognee.infrastructure.databases.provenance import EdgeIdentity

    return [
        EdgeIdentity(source_id=f"s-{index}", target_id=f"t-{index}", relationship_name="relates_to")
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_get_edge_delete_data_chunks_and_seeks():
    """The edge-snapshot query previously sent every identity in ONE statement
    with a cartesian MATCH + WHERE — on a 295k-edge graph it could never finish
    inside the subprocess deadline (hit by rekey_fork_document_ids)."""
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE + 1

    await adapter.get_edge_delete_data(_fake_edge_identities(total))

    assert adapter.query.await_count == 2
    chunk_sizes = [len(call.args[1]["edges"]) for call in adapter.query.await_args_list]
    assert chunk_sizes == [_WRITE_CHUNK_SIZE, 1]
    query = adapter.query.await_args_list[0].args[0]
    assert "MATCH (a:Node {id: e.s})-[r:EDGE]->(b:Node {id: e.t})" in query
    assert "WHERE a.id = e.s" not in query


@pytest.mark.asyncio
async def test_delete_edge_triples_chunks_and_seeks():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE * 2 + 1

    await adapter.delete_edge_triples(_fake_edge_identities(total))

    assert adapter.query.await_count == 3
    query = adapter.query.await_args_list[0].args[0]
    assert "MATCH (a:Node {id: e.s})-[r:EDGE]->(b:Node {id: e.t})" in query
    adapter.checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_edge_provenance_read_and_write_chunk_and_seek():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE + 1

    await adapter._read_edge_provenance(_fake_edge_identities(total))
    read_queries = [call.args[0] for call in adapter.query.await_args_list]
    assert adapter.query.await_count == 2
    assert all("MATCH (a:Node {id: e.s})-[r:EDGE]->(b:Node {id: e.t})" in q for q in read_queries)

    adapter.query.reset_mock()
    batch = [
        {
            "s": f"s-{i}",
            "t": f"t-{i}",
            "rel": "relates_to",
            "refs": [],
            "datasets": [],
            "runs": [],
            "run_refs": [],
        }
        for i in range(total)
    ]
    await adapter._write_edge_provenance(batch)
    assert adapter.query.await_count == 2
    write_query = adapter.query.await_args_list[0].args[0]
    assert "MATCH (a:Node {id: row.s})-[r:EDGE]->(b:Node {id: row.t})" in write_query


@pytest.mark.asyncio
async def test_node_delete_data_and_provenance_chunk_by_id_seek():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE + 1
    ids = [f"node-{i}" for i in range(total)]

    await adapter.get_node_delete_data(ids)
    assert adapter.query.await_count == 2
    assert "MATCH (n:Node {id: nid})" in adapter.query.await_args_list[0].args[0]

    adapter.query.reset_mock()
    await adapter._read_node_provenance(ids)
    assert adapter.query.await_count == 2

    adapter.query.reset_mock()
    batch = [
        {"id": f"node-{i}", "refs": [], "datasets": [], "runs": [], "run_refs": []}
        for i in range(total)
    ]
    await adapter._write_node_provenance(batch)
    assert adapter.query.await_count == 2
    assert "MATCH (n:Node {id: row.id})" in adapter.query.await_args_list[0].args[0]
