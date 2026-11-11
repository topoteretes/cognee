"""Deletion-driven ledger tombstones: one per data item, and the delete wiring."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import cognee.modules.graph.methods.delete_data_nodes_and_edges
import cognee.modules.graph.methods.delete_dataset_nodes_and_edges
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.unified.provenance_delete_planner import (
    SourceRefRemovalResult,
)
from cognee.modules.provenance import storage
from cognee.modules.provenance.tombstones import (
    agent_id_for,
    dataset_id_from_ledger_key,
    ledger_data_item_key,
    ledger_edge_key,
    ledger_node_key,
    tombstone_data_item,
    tombstone_dataset,
    tombstone_pipeline_run,
)

# The package __init__ re-exports the functions under their module names.
ddne_module = sys.modules["cognee.modules.graph.methods.delete_data_nodes_and_edges"]
ddsne_module = sys.modules["cognee.modules.graph.methods.delete_dataset_nodes_and_edges"]

pytestmark = pytest.mark.asyncio

USER = SimpleNamespace(id=uuid4(), email="u@x.io")


async def _seed_items(manager, dataset_id, data_ids, run_id="run-1"):
    """Write what record_provenance writes: one row per data item."""
    batch = manager.batch()
    for data_id in data_ids:
        batch.track_data_item(
            ledger_data_item_key(dataset_id, data_id),
            content_hash="hash-1",
            bundle_id=run_id,
            source_ref_key=make_source_ref_key(dataset_id, data_id),
        )
    await batch.commit()


async def _item(manager, dataset_id, data_id):
    return await manager.get_provenance(ledger_data_item_key(dataset_id, data_id))


def _ledger_down(monkeypatch):
    async def explode(write_fns):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(storage, "append_chained_many", explode)


class TestKeys:
    def test_keys_match_record_provenance_namespacing(self):
        dataset_id, data_id = uuid4(), uuid4()
        assert ledger_node_key(dataset_id, "n1") == f"{dataset_id}:n1"
        assert ledger_edge_key(dataset_id, "a", "b", "knows") == (
            f"rel:{dataset_id}:a:knows:{dataset_id}:b"
        )
        assert ledger_data_item_key(dataset_id, data_id) == f"{dataset_id}:data_item:{data_id}"

    def test_every_key_names_its_dataset(self):
        dataset_id = uuid4()
        for key in (
            ledger_node_key(dataset_id, "n1"),
            ledger_edge_key(dataset_id, "a", "b", "knows"),
            ledger_data_item_key(dataset_id, uuid4()),
        ):
            assert dataset_id_from_ledger_key(key) == dataset_id

    def test_rows_are_attributed_to_the_user_id_never_the_email(self):
        user_id = uuid4()
        assert agent_id_for(SimpleNamespace(email="u@x.io", id=user_id)) == str(user_id)
        assert agent_id_for(None) == "cognee"


class TestTombstoneDataItem:
    async def test_retracts_that_item_only(self, manager):
        dataset_id, deleted, kept = uuid4(), uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [deleted, kept])

        assert await tombstone_data_item(dataset_id, deleted, user=USER) == 1

        tombstone = await _item(manager, dataset_id, deleted)
        assert tombstone["invalidated"] is True
        assert tombstone["invalidation_reason"] == "data_deleted"
        assert tombstone["invalidated_by"] == str(USER.id)
        assert (await _item(manager, dataset_id, kept))["invalidated"] is False
        assert (await manager.get_statistics())["invalidated_count"] == 1
        assert (await manager.verify_chain())["valid"] is True

    async def test_repeat_and_unrecorded_items_write_nothing(self, manager):
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [data_id])
        await tombstone_data_item(dataset_id, data_id, user=USER)
        total = (await manager.get_statistics())["total_entries"]

        assert await tombstone_data_item(dataset_id, data_id, user=USER) == 0
        assert await tombstone_data_item(dataset_id, uuid4(), user=USER) == 0
        assert (await manager.get_statistics())["total_entries"] == total

    async def test_same_data_id_in_another_dataset_is_untouched(self, manager):
        dataset_a, dataset_b, data_id = uuid4(), uuid4(), uuid4()
        await _seed_items(manager, dataset_a, [data_id])
        await _seed_items(manager, dataset_b, [data_id])

        await tombstone_data_item(dataset_a, data_id, user=USER)

        assert (await _item(manager, dataset_a, data_id))["invalidated"] is True
        assert (await _item(manager, dataset_b, data_id))["invalidated"] is False

    async def test_a_later_write_brings_the_item_back(self, manager):
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [data_id])
        await tombstone_data_item(dataset_id, data_id, user=USER, reason="memory_dropped")

        await _seed_items(manager, dataset_id, [data_id], run_id="run-2")

        current = await _item(manager, dataset_id, data_id)
        assert current["invalidated"] is False and current["bundle_id"] == "run-2"
        history = await manager.revision_history(ledger_data_item_key(dataset_id, data_id))
        assert [bool(version.get("invalidated")) for version in history] == [False, True, False]

    async def test_a_tombstone_that_cannot_be_written_raises(self, manager, monkeypatch):
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [data_id])
        _ledger_down(monkeypatch)

        with pytest.raises(RuntimeError, match="ledger down"):
            await tombstone_data_item(dataset_id, data_id, user=USER)
        assert (await _item(manager, dataset_id, data_id))["invalidated"] is False


class TestTombstoneDataset:
    async def test_retracts_every_item_of_that_dataset_only(self, manager):
        dataset_a, dataset_b = uuid4(), uuid4()
        items_a, item_b = [uuid4(), uuid4()], uuid4()
        await _seed_items(manager, dataset_a, items_a)
        await _seed_items(manager, dataset_b, [item_b])

        assert await tombstone_dataset(dataset_a, user=USER) == 2

        for data_id in items_a:
            tombstone = await _item(manager, dataset_a, data_id)
            assert tombstone["invalidated"] is True
            assert tombstone["invalidation_reason"] == "dataset_deleted"
        assert (await _item(manager, dataset_b, item_b))["invalidated"] is False
        assert await tombstone_dataset(dataset_a, user=USER) == 0
        assert (await manager.verify_chain())["valid"] is True

    async def test_a_tombstone_that_cannot_be_written_raises(self, manager, monkeypatch):
        dataset_id = uuid4()
        await _seed_items(manager, dataset_id, [uuid4()])
        _ledger_down(monkeypatch)

        with pytest.raises(RuntimeError, match="ledger down"):
            await tombstone_dataset(dataset_id, user=USER)


class TestTombstonePipelineRun:
    async def test_retracts_the_items_the_run_gave_memory_to(self, manager):
        dataset_id, first, second, other = uuid4(), uuid4(), uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [first, second], run_id="rolled-back")
        await _seed_items(manager, dataset_id, [other], run_id="other-run")

        assert await tombstone_pipeline_run(dataset_id, "rolled-back", user=USER) == 2

        for data_id in (first, second):
            tombstone = await _item(manager, dataset_id, data_id)
            assert tombstone["invalidated"] is True
            assert tombstone["invalidation_reason"] == "pipeline_run_rolled_back"
            assert tombstone["metadata"]["rolled_back_pipeline_run_id"] == "rolled-back"
        assert (await _item(manager, dataset_id, other))["invalidated"] is False

    async def test_completed_items_a_recovery_keeps_stay_live(self, manager):
        dataset_id, kept, unfinished = uuid4(), uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [kept, unfinished], run_id="run")

        written = await tombstone_pipeline_run(dataset_id, "run", user=USER, keep_data_ids=[kept])

        assert written == 1
        assert (await _item(manager, dataset_id, kept))["invalidated"] is False
        assert (await _item(manager, dataset_id, unfinished))["invalidated"] is True

    async def test_item_that_had_memory_before_the_run_stays_live(self, manager):
        # The indexing run stores the chunks first; a cognify run that fails
        # afterwards is rolled back by run id and leaves them in the graph.
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [data_id], run_id="indexing-run")
        await _seed_items(manager, dataset_id, [data_id], run_id="cognify-run")

        assert await tombstone_pipeline_run(dataset_id, "cognify-run", user=USER) == 0
        assert (await _item(manager, dataset_id, data_id))["invalidated"] is False

    async def test_item_the_run_brought_back_is_retracted(self, manager):
        dataset_id, data_id = uuid4(), uuid4()
        await _seed_items(manager, dataset_id, [data_id], run_id="first-run")
        await tombstone_data_item(dataset_id, data_id, user=USER)
        await _seed_items(manager, dataset_id, [data_id], run_id="second-run")

        assert await tombstone_pipeline_run(dataset_id, "second-run", user=USER) == 1
        assert (await _item(manager, dataset_id, data_id))["invalidated"] is True

    async def test_unknown_run_is_noop(self, manager):
        dataset_id = uuid4()
        await _seed_items(manager, dataset_id, [uuid4()])
        assert await tombstone_pipeline_run(dataset_id, "no-such-run", user=USER) == 0


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
    """The graph delete choke points retract the data item, whatever they removed."""

    def _provenance_path(self, dataset_id, removal):
        return (
            patch.object(ddne_module, "get_user", AsyncMock(return_value=USER)),
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
        )

    async def test_delete_data_graph_provenance_path_tombstones(self):
        dataset_id, data_id = uuid4(), uuid4()
        removal = SourceRefRemovalResult(deleted_node_ids=["n1"])
        user_patch, dataset_patch, delete_patch = self._provenance_path(dataset_id, removal)

        with (
            user_patch,
            dataset_patch,
            delete_patch,
            patch.object(ddne_module, "tombstone_data_item", AsyncMock()) as tombstone,
        ):
            result = await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, USER.id)

        tombstone.assert_awaited_once_with(dataset_id, data_id, user=USER)
        assert result.node_ids == {"n1"}

    async def test_a_failed_tombstone_fails_the_delete(self):
        dataset_id, data_id = uuid4(), uuid4()
        user_patch, dataset_patch, delete_patch = self._provenance_path(
            dataset_id, SourceRefRemovalResult()
        )

        with (
            user_patch,
            dataset_patch,
            delete_patch,
            patch.object(
                ddne_module,
                "tombstone_data_item",
                AsyncMock(side_effect=RuntimeError("ledger down")),
            ),
            pytest.raises(RuntimeError, match="ledger down"),
        ):
            await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, USER.id)

    async def test_delete_data_ledger_path_tombstones(self):
        dataset_id, data_id = uuid4(), uuid4()

        with (
            patch.object(ddne_module, "get_user", AsyncMock(return_value=USER)),
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
            patch.object(ddne_module, "tombstone_data_item", AsyncMock()) as tombstone,
        ):
            await ddne_module.delete_data_nodes_and_edges(dataset_id, data_id, USER.id)

        tombstone.assert_awaited_once_with(dataset_id, data_id, user=USER)

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
