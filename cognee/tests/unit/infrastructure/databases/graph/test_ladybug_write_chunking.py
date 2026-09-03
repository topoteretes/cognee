"""Bulk node/edge writes must be chunked so no single statement can run past
the subprocess engine's per-call deadline on large graphs (COG: ladybug
ingestion of e.g. 30k-fact code graphs previously sent one statement for all
rows and could never finish).

Each chunk is an upsert spelled out as probe / CREATE-missing / SET-existing
rather than one MERGE, because Kuzu never plans MERGE through the primary-key
index (COG-6216). The plans themselves are asserted against a real database in
``test_ladybug_write_query_plans.py``; these tests cover chunking and the
statement shapes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import (
    LadybugAdapter,
    _WRITE_CHUNK_SIZE,
)


def _adapter_with_mocked_writes(existing_rows=None):
    """Adapter whose queries are mocked. ``existing_rows`` is what the
    existing-row probe returns, so a test can drive the CREATE path (default:
    nothing exists yet) or the SET path.
    """
    adapter = object.__new__(LadybugAdapter)
    adapter.query = AsyncMock(return_value=existing_rows or [])
    adapter.checkpoint = AsyncMock()
    adapter._bulk_write_lock = asyncio.Lock()
    return adapter


def _write_calls(adapter, param_key):
    """Query calls carrying ``param_key`` — i.e. the writes, not the probe."""
    return [call for call in adapter.query.await_args_list if param_key in call.args[1]]


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

    # Nothing exists yet, so every chunk is probe + CREATE and none update.
    expected = [_WRITE_CHUNK_SIZE, _WRITE_CHUNK_SIZE, 1]
    assert [len(call.args[1]["ids"]) for call in _write_calls(adapter, "ids")] == expected
    assert [len(call.args[1]["nodes"]) for call in _write_calls(adapter, "nodes")] == expected
    adapter.checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_nodes_small_batch_is_one_probe_and_one_write():
    adapter = _adapter_with_mocked_writes()

    await adapter.add_nodes(_fake_nodes(5))

    assert len(_write_calls(adapter, "ids")) == 1
    assert len(_write_calls(adapter, "nodes")) == 1


@pytest.mark.asyncio
async def test_add_nodes_updates_rows_the_probe_found():
    """Ids the probe returns take the SET path. Sending them to CREATE would
    raise a primary-key violation on every re-ingest."""
    adapter = _adapter_with_mocked_writes(existing_rows=[("node-0",), ("node-1",)])

    await adapter.add_nodes(_fake_nodes(3))

    by_kind = {
        ("CREATE" if "CREATE (n:Node" in call.args[0] else "SET"): [
            node["id"] for node in call.args[1]["nodes"]
        ]
        for call in _write_calls(adapter, "nodes")
    }
    assert by_kind["SET"] == ["node-0", "node-1"]
    assert by_kind["CREATE"] == ["node-2"]


@pytest.mark.asyncio
async def test_add_nodes_deduplicates_ids_within_a_batch():
    """MERGE absorbed repeated ids inside one UNWIND; CREATE would raise."""
    adapter = _adapter_with_mocked_writes()
    duplicated = [SimpleNamespace(id="node-0", name=f"n{i}", type="Node") for i in range(3)]

    await adapter.add_nodes(duplicated)

    assert [len(call.args[1]["nodes"]) for call in _write_calls(adapter, "nodes")] == [1]


@pytest.mark.asyncio
async def test_add_edges_chunks_large_batches():
    adapter = _adapter_with_mocked_writes()
    total = _WRITE_CHUNK_SIZE + 1

    await adapter.add_edges(_fake_edges(total))

    assert [len(call.args[1]["edges"]) for call in _write_calls(adapter, "edges")] == [
        _WRITE_CHUNK_SIZE,
        1,
    ]
    adapter.checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_edges_deduplicates_within_a_batch():
    adapter = _adapter_with_mocked_writes()

    await adapter.add_edges(_fake_edges(1) * 3)

    assert [len(call.args[1]["edges"]) for call in _write_calls(adapter, "edges")] == [1]


@pytest.mark.asyncio
async def test_add_edges_stringifies_endpoint_ids():
    """Endpoint ids arrive as UUIDs. The probe compares them against ids read
    back from the STRING id column, so unless they are stringified every edge
    looks new and gets written again on each ingest."""
    import uuid

    adapter = _adapter_with_mocked_writes()
    source, target = uuid.uuid4(), uuid.uuid4()

    await adapter.add_edges([(source, target, "relates_to", {})])

    written = _write_calls(adapter, "edges")[0].args[1]["edges"][0]
    assert written["from_id"] == str(source)
    assert written["to_id"] == str(target)


@pytest.mark.asyncio
async def test_add_edges_matches_endpoints_in_separate_clauses():
    """Both endpoints in ONE comma-separated MATCH plans as a cartesian product
    of two table scans; separate MATCH clauses plan as two index seeks.

    COG-6185 recorded that separate clauses segfault ladybug 0.19.x mid-write,
    but that was with MERGE, which these statements no longer use — CREATE and
    SET were exercised on 0.19.0 up to 40k edges through the subprocess worker
    without a SIGSEGV. Keeping MERGE out of the split shape is the invariant.
    """
    adapter = _adapter_with_mocked_writes()

    await adapter.add_edges(_fake_edges(1))

    query = _write_calls(adapter, "edges")[0].args[0]
    assert "MATCH (from:Node {id: edge.from_id})\nMATCH (to:Node {id: edge.to_id})" in query
    assert "MATCH (from:Node {id: edge.from_id}), (to:Node {id: edge.to_id})" not in query
    assert "MERGE" not in query


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
