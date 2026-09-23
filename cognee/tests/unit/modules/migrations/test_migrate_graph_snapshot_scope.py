"""_migrate_graph must snapshot provenance only for edges whose snapshot is
consumed (remapped edges + survivors incident to an affected neighbor) —
snapshotting the whole graph made large fork re-keys unable to finish."""

from unittest.mock import AsyncMock, patch

import pytest

import cognee.modules.migrations.versions.namespace_entity_type_node_ids as ns


@pytest.mark.asyncio
async def test_migrate_graph_snapshots_only_consumable_edges():
    graph_engine = AsyncMock()
    id_map = {"a": "a2"}
    properties_by_id = {"a": {"id": "a", "name": "a", "type": "Node"}}
    edges = [
        ("a", "b", "contains", {}),  # remapped (touches id_map)
        ("b", "c", "relates_to", {}),  # survivor at risk (touches neighbor b)
        ("c", "d", "relates_to", {}),  # untouched — must NOT be snapshotted
    ]

    with patch.object(
        ns, "_snapshot_graph_provenance", new=AsyncMock(return_value=({}, {}))
    ) as snapshot:
        await ns._migrate_graph(graph_engine, id_map, properties_by_id, edges)

    snapshot.assert_awaited_once()
    _, node_ids, snapshot_edges = snapshot.await_args.args
    assert node_ids == ["a"]
    assert ("a", "b", "contains", {}) in snapshot_edges
    assert ("b", "c", "relates_to", {}) in snapshot_edges
    assert ("c", "d", "relates_to", {}) not in snapshot_edges


@pytest.mark.asyncio
async def test_restore_provenance_bulk_batches_by_profile():
    """One attach per (provenance profile, key) — not one per artifact. The
    per-artifact loop made a 100k-edge restore take hours of sequential
    subprocess round-trips."""
    from types import SimpleNamespace

    attach = AsyncMock()
    shared = SimpleNamespace(
        source_ref_keys=["key-A"],
        source_run_refs=["source_run_ref:v1:00000000-0000-0000-0000-000000000001:key-A"],
    )
    items = [(f"edge-{i}", shared) for i in range(1000)]
    items.append(("edge-none", None))  # snapshot-less artifacts are skipped

    await ns._restore_provenance_bulk(attach, items)

    attach.assert_awaited_once()
    artifacts, keys, _run_id = attach.await_args.args
    assert len(artifacts) == 1000
    assert keys == ["key-A"]


@pytest.mark.asyncio
async def test_rekey_provenance_moves_are_batched_by_run():
    """_rekey_graph_provenance must attach refs per run-id group, not one
    artifact at a time — the per-item loop's checkpoint churn could exhaust
    kuzu's max_db_size on 100k-node graphs."""
    from types import SimpleNamespace
    from uuid import uuid4

    import cognee.modules.migrations.versions.rekey_fork_document_ids as rk
    from cognee.infrastructure.databases.provenance import make_source_ref_key

    old_id, new_id, dataset_id = str(uuid4()), str(uuid4()), str(uuid4())
    old_key = make_source_ref_key(dataset_id, __import__("uuid").UUID(old_id))
    run_uuid = "00000000-0000-0000-0000-000000000001"
    snapshot = SimpleNamespace(source_run_refs=[f"source_run_ref:v1:{run_uuid}:{old_key}"])

    engine = AsyncMock()
    engine.find_nodes_by_source_ref.return_value = ["n1", "n2", "n3"]
    engine.find_edges_by_source_ref.return_value = []
    engine.get_node_delete_data.return_value = {n: snapshot for n in ("n1", "n2", "n3")}
    # The v1 sweep this test asserts on is followed by the chunk-scoped (v2)
    # sweep, which reads the dataset's ref maps; no v2 refs in this fixture.
    engine.find_node_source_refs_by_dataset.return_value = {}
    engine.find_edge_source_refs_by_dataset.return_value = {}

    with patch.object(rk, "stores_provenance_in_graph", new=AsyncMock(return_value=True)):
        await rk._rekey_graph_provenance(engine, [(old_id, new_id, dataset_id)])

    engine.attach_node_source_refs.assert_awaited_once()
    ids, _keys, run_id = engine.attach_node_source_refs.await_args.args
    assert sorted(ids) == ["n1", "n2", "n3"]
    assert run_id == run_uuid
