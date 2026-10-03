"""Deletion-driven ledger tombstones: key mapping, sweeps, and delete wiring."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import cognee.modules.graph.methods.delete_data_nodes_and_edges
import cognee.modules.graph.methods.delete_dataset_nodes_and_edges
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.unified.provenance_delete_planner import (
    SourceRefRemovalResult,
)
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.provenance import storage
from cognee.modules.provenance.tombstones import (
    agent_id_for,
    ledger_edge_key,
    ledger_node_key,
    tombstone_dataset,
    tombstone_deleted_elements,
    tombstone_pipeline_run,
)

# The package __init__ re-exports the functions under their module names.
ddne_module = sys.modules["cognee.modules.graph.methods.delete_data_nodes_and_edges"]
ddsne_module = sys.modules["cognee.modules.graph.methods.delete_dataset_nodes_and_edges"]

pytestmark = pytest.mark.asyncio


async def _seed_dataset(manager, dataset_id, node_ids, edges=()):
    """Write a document-less slice of what record_provenance would write."""
    batch = manager.batch()
    for node_id in node_ids:
        batch.track_entity(ledger_node_key(dataset_id, node_id), source="doc", entity_type="entity")
    for source, target, name in edges:
        batch.track_relationship(
            ledger_edge_key(dataset_id, source, target, name),
            source="doc",
            used_entities=[
                ledger_node_key(dataset_id, source),
                ledger_node_key(dataset_id, target),
            ],
        )
    await batch.commit()


class TestKeys:
    def test_keys_match_record_provenance_namespacing(self):
        dataset_id = uuid4()
        assert ledger_node_key(dataset_id, "n1") == f"{dataset_id}:n1"
        assert ledger_edge_key(dataset_id, "a", "b", "knows") == (
            f"rel:{dataset_id}:a:knows:{dataset_id}:b"
        )

    def test_agent_id_precedence(self):
        user_id = uuid4()
        assert agent_id_for(SimpleNamespace(email="u@x.io", id=user_id)) == "u@x.io"
        assert agent_id_for(SimpleNamespace(email=None, id=user_id)) == str(user_id)
        assert agent_id_for(None) == "cognee"

    def test_deleted_graph_elements_carry_edge_keys(self):
        result = SourceRefRemovalResult(
            deleted_node_ids=["n1"], deleted_edges=[EdgeIdentity("n1", "n2", "knows")]
        )
        deleted = DeletedGraphElements.from_source_ref_removal(result)
        assert deleted.node_ids == {"n1"}
        assert deleted.edge_keys == {("n1", "n2", "knows")}
        assert len(deleted.edge_ids) == 1

        other = DeletedGraphElements(node_ids={"n3"}, edge_keys={("n3", "n4", "x")})
        deleted.merge(other)
        assert deleted.edge_keys == {("n1", "n2", "knows"), ("n3", "n4", "x")}


class TestTombstoneDeletedElements:
    async def test_tombstones_only_live_tracked_elements(self, manager):
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_dataset(manager, dataset_id, ["n1", "n2", "n3"], [("n1", "n2", "knows")])

        deleted = DeletedGraphElements(
            node_ids={"n1", "n2", "never-tracked"},
            edge_keys={("n1", "n2", "knows"), ("n2", "n3", "untracked-edge")},
        )
        user = SimpleNamespace(email="auditor@x.io", id=uuid4())
        written = await tombstone_deleted_elements(dataset_id, deleted, user=user, data_id=data_id)
        assert written == 3  # n1, n2, rel — untracked keys are skipped, not raised

        for raw in ("n1", "n2"):
            stored = await manager.get_provenance(ledger_node_key(dataset_id, raw))
            assert stored["invalidated"] is True
            assert stored["invalidated_by"] == "auditor@x.io"
            assert stored["invalidation_reason"] == "data_deleted"
            assert stored["metadata"]["deleted_data_id"] == str(data_id)
            assert stored["metadata"]["deleted_dataset_id"] == str(dataset_id)
        rel = await manager.get_provenance(ledger_edge_key(dataset_id, "n1", "n2", "knows"))
        assert rel["invalidated"] is True
        # Survivor untouched.
        assert (await manager.get_provenance(ledger_node_key(dataset_id, "n3")))[
            "invalidated"
        ] is False
        assert (await manager.verify_chain())["valid"] is True
        assert (await manager.check())["valid"] is True

    async def test_repeat_is_noop(self, manager):
        dataset_id = uuid4()
        await _seed_dataset(manager, dataset_id, ["n1"])
        deleted = DeletedGraphElements(node_ids={"n1"})
        assert await tombstone_deleted_elements(dataset_id, deleted) == 1
        assert await tombstone_deleted_elements(dataset_id, deleted) == 0
        assert (await manager.get_statistics())["total_entries"] == 2  # live + one archive

    async def test_empty_and_none_inputs(self, manager):
        assert await tombstone_deleted_elements(uuid4(), None) == 0
        assert await tombstone_deleted_elements(uuid4(), DeletedGraphElements()) == 0

    async def test_never_raises(self, manager, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(storage, "retrieve_live_ids", explode)
        deleted = DeletedGraphElements(node_ids={"n1"})
        assert await tombstone_deleted_elements(uuid4(), deleted) == 0


class TestTombstoneDataset:
    async def test_sweeps_namespace_only(self, manager):
        dataset_id, other_dataset = uuid4(), uuid4()
        await _seed_dataset(manager, dataset_id, ["n1", "n2"], [("n1", "n2", "knows")])
        await _seed_dataset(manager, other_dataset, ["n1"], [("n1", "n1", "self")])
        # A versioned entity: its archive shares the prefix but must not be
        # tombstoned (it is history, not a live assertion).
        await manager.track_entity(ledger_node_key(dataset_id, "n1"), source="doc-2")

        written = await tombstone_dataset(dataset_id, user=SimpleNamespace(email="u@x.io"))
        assert written == 3  # n1 (live), n2, rel

        for raw in ("n1", "n2"):
            stored = await manager.get_provenance(ledger_node_key(dataset_id, raw))
            assert stored["invalidated"] is True
            assert stored["invalidation_reason"] == "dataset_deleted"
        assert (await manager.get_provenance(ledger_edge_key(dataset_id, "n1", "n2", "knows")))[
            "invalidated"
        ] is True

        # Other dataset's rows are live; archives are untouched.
        assert (await manager.get_provenance(ledger_node_key(other_dataset, "n1")))[
            "invalidated"
        ] is False
        archives = [
            entry
            for entry in await storage.retrieve_all()
            if ":v:" in entry.entity_id and entry.entity_id.startswith(f"{dataset_id}:n1")
        ]
        # One archive from the re-track (live, pre-version) + one from the
        # tombstone (the pre-invalidation state); neither is itself a tombstone.
        assert len(archives) == 2
        assert all(archive.invalidated is False for archive in archives)
        assert (await manager.verify_chain())["valid"] is True

        # Second sweep finds nothing live.
        assert await tombstone_dataset(dataset_id) == 0

    async def test_resurrection_after_dataset_sweep(self, manager):
        """forget(memory_only=True) then cognify again: same ids, live again."""
        dataset_id = uuid4()
        await _seed_dataset(manager, dataset_id, ["n1"], [("n1", "n1", "self")])
        assert await tombstone_dataset(dataset_id) == 2
        await _seed_dataset(manager, dataset_id, ["n1"], [("n1", "n1", "self")])
        assert (await manager.get_provenance(ledger_node_key(dataset_id, "n1")))[
            "invalidated"
        ] is False
        assert (await manager.get_provenance(ledger_edge_key(dataset_id, "n1", "n1", "self")))[
            "invalidated"
        ] is False
        assert (await manager.verify_chain())["valid"] is True

    async def test_never_raises(self, manager, monkeypatch):
        async def explode(*args, **kwargs):
            raise RuntimeError("db down")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(storage, "iter_live_with_prefix", explode)
        assert await tombstone_dataset(uuid4()) == 0


class TestTombstonePipelineRun:
    async def test_retracts_only_what_the_run_first_asserted(self, manager):
        from cognee.infrastructure.databases.provenance import make_source_ref_key

        dataset_id, run_a, run_b = uuid4(), uuid4(), uuid4()
        doc_x, doc_kept = uuid4(), uuid4()
        ref_x = make_source_ref_key(dataset_id, doc_x)
        ref_kept = make_source_ref_key(dataset_id, doc_kept)

        # Run A asserts n1. Run B re-mentions n1 (versions it), first-asserts
        # n2 (doc_x), n3 (doc_kept) and the edge n2->n3.
        await manager.track_entity(
            ledger_node_key(dataset_id, "n1"),
            source="a",
            bundle_id=str(run_a),
            source_ref_key=ref_x,
        )
        batch = manager.batch()
        batch.track_entity(
            ledger_node_key(dataset_id, "n1"),
            source="b",
            bundle_id=str(run_b),
            source_ref_key=ref_x,
        )
        batch.track_entity(
            ledger_node_key(dataset_id, "n2"),
            source="b",
            bundle_id=str(run_b),
            source_ref_key=ref_x,
        )
        batch.track_entity(
            ledger_node_key(dataset_id, "n3"),
            source="b",
            bundle_id=str(run_b),
            source_ref_key=ref_kept,
        )
        batch.track_relationship(
            ledger_edge_key(dataset_id, "n2", "n3", "knows"),
            source="b",
            bundle_id=str(run_b),
            source_ref_key=ref_x,
        )
        await batch.commit()

        written = await tombstone_pipeline_run(dataset_id, run_b, keep_data_ids={doc_kept})
        assert written == 2  # n2 and the edge; n1 survives (re-mention), n3 is kept

        get = manager.get_provenance
        assert (await get(ledger_node_key(dataset_id, "n1")))["invalidated"] is False
        assert (await get(ledger_node_key(dataset_id, "n2")))["invalidated"] is True
        assert (await get(ledger_node_key(dataset_id, "n3")))["invalidated"] is False
        edge = await get(ledger_edge_key(dataset_id, "n2", "n3", "knows"))
        assert edge["invalidated"] is True
        assert edge["invalidation_reason"] == "pipeline_run_rolled_back"
        assert edge["metadata"]["rolled_back_pipeline_run_id"] == str(run_b)
        assert (await manager.verify_chain())["valid"] is True
        # Idempotent.
        assert await tombstone_pipeline_run(dataset_id, run_b, keep_data_ids={doc_kept}) == 0

    async def test_unknown_run_is_noop(self, manager):
        assert await tombstone_pipeline_run(uuid4(), uuid4()) == 0


class TestDatasetScopedReads:
    """Audit reads narrowed to one dataset's key namespace."""

    async def _seed(self, manager):
        ds_a, ds_b = uuid4(), uuid4()
        batch = manager.batch()
        batch.track_entity(ledger_node_key(ds_a, "n1"), source="a")
        batch.track_entity(ledger_node_key(ds_a, "n2"), source="a")
        batch.track_relationship(ledger_edge_key(ds_a, "n1", "n2", "knows"), source="a")
        batch.track_entity(ledger_node_key(ds_b, "n1"), source="b")
        await batch.commit()
        # Tombstone + resurrect n2 in A: leaves an archived tombstone copy behind.
        await manager.invalidate(ledger_node_key(ds_a, "n2"), "auditor", reason="x")
        await manager.track_entity(ledger_node_key(ds_a, "n2"), source="a")
        # A live tombstone in A.
        await manager.invalidate(ledger_node_key(ds_a, "n1"), "auditor", reason="y")
        return ds_a, ds_b

    async def test_statistics_split_live_from_archived_and_count_live_tombstones(self, manager):
        ds_a, ds_b = await self._seed(manager)

        stats_a = await manager.get_statistics(dataset_id=ds_a)
        assert stats_a["dataset_id"] == str(ds_a)
        # n1, n2, edge live; archives: n2 tombstone, n2 pre-tombstone, n1 pre-tombstone.
        assert stats_a["live_entries"] == 3
        assert stats_a["archived_entries"] == 3
        assert stats_a["total_entries"] == 6
        assert stats_a["invalidated_count"] == 1  # n1 only — n2's tombstone is archived

        stats_b = await manager.get_statistics(dataset_id=ds_b)
        assert (stats_b["live_entries"], stats_b["archived_entries"]) == (1, 0)
        assert stats_b["invalidated_count"] == 0

        everything = await manager.get_statistics()
        assert everything["total_entries"] == 7
        assert everything["invalidated_count"] == 1
        assert everything["dataset_id"] is None

    async def test_check_and_verify_are_scoped(self, manager):
        ds_a, ds_b = await self._seed(manager)

        check_a = await manager.check(dataset_id=ds_a)
        assert check_a["valid"] is True
        assert check_a["total_entries"] == 6
        assert check_a["invalidated_count"] == 1
        assert (await manager.check(dataset_id=ds_b))["total_entries"] == 1

        verify_a = await manager.verify_chain(dataset_id=ds_a)
        assert verify_a["valid"] is True
        assert verify_a["total_entries"] == 6
        assert verify_a["dataset_id"] == str(ds_a)
        assert (await manager.verify_chain(dataset_id=ds_b))["total_entries"] == 1
        assert (await manager.verify_chain(dataset_id=uuid4()))["total_entries"] == 0

    async def test_scoped_verify_detects_tampering(self, manager):
        from sqlalchemy import update

        from cognee.modules.provenance.models import ProvenanceEntryRow

        ds_a, _ = await self._seed(manager)
        async with storage.get_async_session() as session, session.begin():
            await session.execute(
                update(ProvenanceEntryRow)
                .where(ProvenanceEntryRow.entity_id == ledger_node_key(ds_a, "n1"))
                .values(source_document="forged")
            )
        verify_a = await manager.verify_chain(dataset_id=ds_a)
        assert verify_a["valid"] is False
        reasons = {link["reason"] for link in verify_a["broken_links"]}
        assert reasons == {"checksum_mismatch"}

    async def test_export_streams_in_sequence_order(self, manager):
        ds_a, _ = await self._seed(manager)
        rows = [row async for row in manager.export(dataset_id=ds_a)]
        assert len(rows) == 6
        assert [row["sequence_id"] for row in rows] == sorted(row["sequence_id"] for row in rows)
        current_only = [
            row async for row in manager.export(dataset_id=ds_a, include_archived=False)
        ]
        assert {row["entity_id"] for row in current_only} == {
            ledger_node_key(ds_a, "n1"),
            ledger_node_key(ds_a, "n2"),
            ledger_edge_key(ds_a, "n1", "n2", "knows"),
        }


class TestDeleteWiring:
    """The graph delete choke points hand their results to the ledger."""

    async def test_delete_data_graph_provenance_path_tombstones(self):
        dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()
        user = SimpleNamespace(id=user_id, email="u@x.io")
        removal = SourceRefRemovalResult(
            deleted_node_ids=["n1"], deleted_edges=[EdgeIdentity("n1", "n2", "knows")]
        )

        with (
            patch.object(ddne_module, "get_user", AsyncMock(return_value=user)),
            patch.object(
                ddne_module,
                "get_authorized_dataset",
                AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
            ),
            patch.object(
                ddne_module,
                "try_delete_data_by_graph_provenance",
                AsyncMock(return_value=removal),
            ),
            patch.object(ddne_module, "tombstone_deleted_elements", AsyncMock()) as tombstone,
        ):
            result = await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

        tombstone.assert_awaited_once_with(dataset_id, result, user=user, data_id=data_id)
        assert result.node_ids == {"n1"}
        assert result.edge_keys == {("n1", "n2", "knows")}

    async def test_delete_data_ledger_path_tombstones(self):
        dataset_id, data_id, user_id = uuid4(), uuid4(), uuid4()
        user = SimpleNamespace(id=user_id, email="u@x.io")

        with (
            patch.object(ddne_module, "get_user", AsyncMock(return_value=user)),
            patch.object(
                ddne_module,
                "get_authorized_dataset",
                AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
            ),
            patch.object(
                ddne_module, "try_delete_data_by_graph_provenance", AsyncMock(return_value=None)
            ),
            patch.object(ddne_module, "backend_access_control_enabled", lambda: False),
            patch.object(ddne_module, "get_global_data_related_nodes", AsyncMock(return_value=[])),
            patch.object(
                ddne_module, "get_shared_slugs_losing_dataset_anchor", AsyncMock(return_value=[])
            ),
            patch.object(ddne_module, "delete_data_related_nodes", AsyncMock()),
            patch.object(ddne_module, "delete_data_related_edges", AsyncMock()),
            patch.object(ddne_module, "tombstone_deleted_elements", AsyncMock()) as tombstone,
        ):
            result = await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, user_id)

        tombstone.assert_awaited_once_with(dataset_id, result, user=user, data_id=data_id)

    async def test_delete_dataset_graph_provenance_path_tombstones(self):
        dataset_id, user_id = uuid4(), uuid4()
        user = SimpleNamespace(id=user_id, email="u@x.io")
        unified = SimpleNamespace(
            supports_graph_provenance_delete=lambda: True,
            graph=object(),
            delete_by_dataset_id=AsyncMock(),
        )

        with (
            patch.object(ddsne_module, "get_user", AsyncMock(return_value=user)),
            patch.object(
                ddsne_module,
                "get_authorized_dataset",
                AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
            ),
            patch.object(ddsne_module, "get_unified_engine", AsyncMock(return_value=unified)),
            patch.object(ddsne_module, "stores_provenance_in_graph", AsyncMock(return_value=True)),
            patch.object(ddsne_module, "tombstone_dataset", AsyncMock()) as tombstone,
        ):
            await ddsne_module.delete_dataset_nodes_and_edges(dataset_id, user_id)

        tombstone.assert_awaited_once_with(dataset_id, user=user)

    async def test_delete_dataset_ledger_path_tombstones(self):
        dataset_id, user_id = uuid4(), uuid4()
        user = SimpleNamespace(id=user_id, email="u@x.io")
        unified = SimpleNamespace(supports_graph_provenance_delete=lambda: False)

        with (
            patch.object(ddsne_module, "get_user", AsyncMock(return_value=user)),
            patch.object(
                ddsne_module,
                "get_authorized_dataset",
                AsyncMock(return_value=SimpleNamespace(id=dataset_id)),
            ),
            patch.object(ddsne_module, "get_unified_engine", AsyncMock(return_value=unified)),
            patch.object(ddsne_module, "backend_access_control_enabled", lambda: True),
            patch.object(ddsne_module, "get_dataset_related_nodes", AsyncMock(return_value=[])),
            patch.object(ddsne_module, "delete_dataset_related_nodes", AsyncMock()),
            patch.object(ddsne_module, "delete_dataset_related_edges", AsyncMock()),
            patch.object(ddsne_module, "tombstone_dataset", AsyncMock()) as tombstone,
        ):
            await ddsne_module.delete_dataset_nodes_and_edges(dataset_id, user_id)

        tombstone.assert_awaited_once_with(dataset_id, user=user)

    async def test_incremental_update_tombstones_replaced_chunk_output(self):
        from cognee.api.v1.update import incremental

        dataset_id, data_id = uuid4(), uuid4()
        user = SimpleNamespace(id=uuid4(), email="u@x.io")
        removal = SourceRefRemovalResult(
            deleted_node_ids=["c1", "e1"], deleted_edges=[EdgeIdentity("c1", "e1", "contains")]
        )

        with (
            patch.object(
                incremental, "delete_chunks_incremental", AsyncMock(return_value=removal)
            ) as delete,
            patch.object(incremental, "tombstone_deleted_elements", AsyncMock()) as tombstone,
        ):
            await incremental._retire_replaced_chunks(["c1"], dataset_id, data_id, user)

        delete.assert_awaited_once_with(["c1"], dataset_id, data_id)
        tombstone.assert_awaited_once()
        args, kwargs = tombstone.await_args
        assert args[0] == dataset_id
        assert args[1].node_ids == {"c1", "e1"}
        assert args[1].edge_keys == {("c1", "e1", "contains")}
        assert kwargs == {"user": user, "data_id": data_id, "reason": "chunks_replaced"}

    async def test_incremental_update_skips_ledger_when_nothing_was_deleted(self):
        from cognee.api.v1.update import incremental

        with (
            patch.object(incremental, "delete_chunks_incremental", AsyncMock(return_value=None)),
            patch.object(incremental, "tombstone_deleted_elements", AsyncMock()) as tombstone,
        ):
            await incremental._retire_replaced_chunks([], uuid4(), uuid4(), None)

        tombstone.assert_not_awaited()
