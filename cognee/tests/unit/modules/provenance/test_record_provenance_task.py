"""record_provenance task: the data item row, failure behaviour, and pipeline wiring."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.config import CognifyConfig, get_cognify_config
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.provenance import storage
from cognee.modules.provenance.tombstones import ledger_data_item_key
from cognee.tasks.provenance import record_provenance, record_provenance_at_storage
from cognee.tasks.provenance.record_provenance import record_data_item_content

# Grab the actual module object for patching (the package __init__ re-exports
# the record_provenance FUNCTION, shadowing the submodule for dotted patching).
record_provenance_module = sys.modules["cognee.tasks.provenance.record_provenance"]
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


class Fact(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def _batch():
    """What add_data_points stored; the writer never looks inside it."""
    return [Fact(name="Alice"), Fact(name="Acme")]


def _ctx(dataset_id=None, pipeline_run_id=None, content_hash="hash-1", data_id=None):
    return PipelineContext(
        user=SimpleNamespace(email="user@example.com", id=uuid4()),
        data_item=SimpleNamespace(id=data_id or uuid4(), content_hash=content_hash),
        dataset=SimpleNamespace(id=dataset_id or uuid4()),
        pipeline_run_id=pipeline_run_id or uuid4(),
    )


async def _only_entry(manager):
    [entry] = await storage.retrieve_all()
    return entry


class TestRecordProvenanceTask:
    @pytest.mark.asyncio
    async def test_records_one_row_for_the_data_item_and_nothing_it_produced(self, manager):
        data_points, ctx = _batch(), _ctx()

        assert await record_provenance(data_points, ctx=ctx) is data_points

        entry = await _only_entry(manager)
        run_id = str(ctx.pipeline_run_id)
        assert entry.entity_id == ledger_data_item_key(ctx.dataset.id, ctx.data_item.id)
        assert entry.entity_type == "data_item"
        assert entry.bundle_id == run_id
        assert entry.activity_id == f"cognify:{run_id}"
        assert entry.source_ref_key == make_source_ref_key(ctx.dataset.id, ctx.data_item.id)
        # Ids and the content hash only: no name, no path, no email.
        assert entry.agent_id == str(ctx.user.id)
        assert entry.source_document == ""
        assert entry.metadata == {
            "snapshot": {"fields": {"content_hash": "hash-1"}, "hash": "hash-1"}
        }
        assert (await manager.verify_chain())["valid"] is True
        assert (await manager.check())["valid"] is True

    @pytest.mark.asyncio
    async def test_later_batches_of_the_same_content_write_nothing(self, manager):
        ctx = _ctx()
        await record_provenance(_batch(), ctx=ctx)
        first = await _only_entry(manager)

        # Indexing, extraction and improve all store batches for one item, under
        # different runs. The row stays with the run that first stored for it.
        later = _ctx(dataset_id=ctx.dataset.id, data_id=ctx.data_item.id, pipeline_run_id=uuid4())
        await record_provenance(_batch(), ctx=later)

        entry = await _only_entry(manager)
        assert entry.sequence_id == first.sequence_id
        assert entry.bundle_id == str(ctx.pipeline_run_id)

    @pytest.mark.asyncio
    async def test_later_batches_of_the_same_content_never_take_the_write_lock(
        self, manager, monkeypatch
    ):
        """Nearly every batch stored for an item finds its row already recorded. Those stop
        at a plain read: the serialized write is shared by every writer in the deployment."""
        ctx = _ctx()
        await record_provenance(_batch(), ctx=ctx)

        async def no_write(write_fns):
            raise AssertionError("a recorded data item went through the serialized write")

        monkeypatch.setattr(storage, "append_chained_many", no_write)
        later = _ctx(dataset_id=ctx.dataset.id, data_id=ctx.data_item.id, pipeline_run_id=uuid4())
        await record_provenance(_batch(), ctx=later)
        await record_provenance(_batch(), ctx=later)

    @pytest.mark.asyncio
    async def test_the_read_that_skips_a_write_agrees_with_the_write_rule(self, manager):
        ctx = _ctx()
        key = ledger_data_item_key(ctx.dataset.id, ctx.data_item.id)
        source_ref = make_source_ref_key(ctx.dataset.id, ctx.data_item.id)
        assert await manager.data_item_is_recorded(key, "hash-1", source_ref) is False  # no row

        await record_provenance(_batch(), ctx=ctx)
        assert await manager.data_item_is_recorded(key, "hash-1", source_ref) is True
        assert await manager.data_item_is_recorded(key, "hash-2", source_ref) is False  # an update
        assert await manager.data_item_is_recorded(key, "hash-1", "other-ref") is False

        await manager.invalidate(key, "someone", reason="data_deleted")
        # A tombstoned row is never current: storing for the item again brings it back.
        assert await manager.data_item_is_recorded(key, "hash-1", source_ref) is False

    @pytest.mark.asyncio
    async def test_changed_content_versions_the_row_with_both_hashes(self, manager):
        ctx = _ctx(content_hash="hash-1")
        await record_provenance(_batch(), ctx=ctx)
        changed = _ctx(dataset_id=ctx.dataset.id, data_id=ctx.data_item.id, content_hash="hash-2")
        await record_provenance(_batch(), ctx=changed)

        key = ledger_data_item_key(ctx.dataset.id, ctx.data_item.id)
        history = await manager.revision_history(key)
        assert [version.get("content_hash") for version in history] == ["hash-1", "hash-2"]
        assert history[-1]["delta"] == {"content_hash": ["hash-1", "hash-2"]}
        assert (await manager.verify_chain())["valid"] is True

    @pytest.mark.asyncio
    async def test_dataset_scoping_isolates_tenants(self, manager):
        data_id = uuid4()
        ctx_a, ctx_b = _ctx(data_id=data_id), _ctx(data_id=data_id)
        await record_provenance(_batch(), ctx=ctx_a)
        await record_provenance(_batch(), ctx=ctx_b)

        keys = {entry.entity_id for entry in await storage.retrieve_all()}
        assert keys == {
            ledger_data_item_key(ctx_a.dataset.id, data_id),
            ledger_data_item_key(ctx_b.dataset.id, data_id),
        }

    @pytest.mark.asyncio
    async def test_a_batch_that_names_no_data_item_records_nothing(self, manager):
        await record_provenance(_batch(), ctx=None)
        await record_provenance(_batch(), ctx=PipelineContext())
        await record_provenance(_batch(), ctx=PipelineContext(dataset=SimpleNamespace(id=uuid4())))
        assert (await manager.get_statistics())["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_a_ledger_that_cannot_be_written_raises(self, manager, monkeypatch):
        async def explode(write_fns):
            raise RuntimeError("ledger down")

        monkeypatch.setattr(storage, "append_chained_many", explode)
        with pytest.raises(RuntimeError, match="ledger down"):
            await record_provenance(_batch(), ctx=_ctx())

    @pytest.mark.asyncio
    async def test_empty_input_is_a_noop(self, manager):
        assert await record_provenance([], ctx=_ctx()) == []
        assert (await manager.get_statistics())["total_entries"] == 0


async def _task_names(provenance_flag):
    config = CognifyConfig(provenance_tracking=provenance_flag)
    with patch.object(cognify_module, "get_cognify_config", return_value=config):
        tasks = await get_default_tasks(
            # Non-None config skips the ontology-env branch; explicit chunk_size
            # keeps get_max_chunk_tokens() from being called.
            config={"ontology_config": {"ontology_resolver": None}},
            chunk_size=1024,
        )
    return [task.executable.__name__ for task in tasks]


class TestPipelineWiring:
    def test_flag_defaults_off_and_lands_in_to_dict(self):
        with patch.dict(os.environ, {}, clear=True):
            config = CognifyConfig()
            assert config.provenance_tracking is False
            assert config.to_dict()["provenance_tracking"] is False

    def test_env_var_enables_flag(self):
        with patch.dict(os.environ, {"PROVENANCE_TRACKING": "true"}):
            get_cognify_config.cache_clear()
            assert get_cognify_config().provenance_tracking is True
        get_cognify_config.cache_clear()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag", [False, True])
    async def test_ledger_is_never_a_pipeline_task(self, flag):
        # Recording lives in add_data_points (storage seam); the task list is
        # the same with the flag on or off.
        names = await _task_names(provenance_flag=flag)
        assert "record_provenance" not in names
        assert "add_data_points" in names


class TestStorageSeamHook:
    @pytest.mark.asyncio
    async def test_records_when_flag_on(self, manager, monkeypatch):
        monkeypatch.setattr(record_provenance_module, "provenance_tracking_enabled", lambda: True)
        data_points = _batch()
        assert await record_provenance_at_storage(data_points, ctx=_ctx()) is data_points
        assert (await manager.get_statistics())["total_entries"] == 1

    @pytest.mark.asyncio
    async def test_skips_when_flag_off(self, manager, monkeypatch):
        monkeypatch.setattr(record_provenance_module, "provenance_tracking_enabled", lambda: False)
        await record_provenance_at_storage(_batch(), ctx=_ctx())
        await record_data_item_content(_ctx(), "hash-2")
        assert (await manager.get_statistics())["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_gate_failure_raises(self, monkeypatch):
        def explode():
            raise RuntimeError("settings unreadable")

        monkeypatch.setattr(record_provenance_module, "provenance_tracking_enabled", explode)
        with pytest.raises(RuntimeError, match="settings unreadable"):
            await record_provenance_at_storage(_batch(), ctx=_ctx())

    @pytest.mark.asyncio
    async def test_new_content_recorded_outside_add_data_points(self, manager, monkeypatch):
        # The incremental update stores its chunks under the old Data row and
        # then states the item's new content hash itself.
        monkeypatch.setattr(record_provenance_module, "provenance_tracking_enabled", lambda: True)
        ctx = _ctx(content_hash="hash-1")
        await record_provenance_at_storage(_batch(), ctx=ctx)
        await record_data_item_content(ctx, "hash-2")

        key = ledger_data_item_key(ctx.dataset.id, ctx.data_item.id)
        history = await manager.revision_history(key)
        assert history[-1]["delta"] == {"content_hash": ["hash-1", "hash-2"]}

    @pytest.mark.asyncio
    async def test_add_data_points_calls_the_hook(self):
        adp_module = sys.modules["cognee.tasks.storage.add_data_points"]
        from cognee.tasks.storage.add_data_points import add_data_points

        class Point(DataPoint):
            name: str
            metadata: dict = {"index_fields": ["name"]}

        point = Point(name="p")
        unified = SimpleNamespace(
            graph=SimpleNamespace(add_nodes=AsyncMock(), add_edges=AsyncMock()),
            vector=object(),
            has_capability=lambda *_: False,
        )
        with (
            patch.object(adp_module, "get_unified_engine", AsyncMock(return_value=unified)),
            # Graph-provenance path: no relational upserts, no real user needed.
            patch.object(
                adp_module, "mark_graph_provenance_if_empty", AsyncMock(return_value=True)
            ),
            patch.object(adp_module, "index_data_points", AsyncMock()),
            patch.object(adp_module, "index_graph_edges", AsyncMock()),
            patch.object(adp_module, "capture_graph_provenance", AsyncMock(return_value=0)),
            patch.object(adp_module, "record_provenance_at_storage", AsyncMock()) as hook,
        ):
            ctx = _ctx()
            await add_data_points([point], ctx=ctx)
        hook.assert_awaited_once_with([point], ctx)
