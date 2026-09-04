"""record_provenance task: ledger writes, degradation, and pipeline wiring."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from cognee.api.v1.cognify.cognify import get_default_tasks
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.engine import DataPoint
from cognee.modules.cognify.config import CognifyConfig, get_cognify_config
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.modules.provenance import storage
from cognee.tasks.provenance import record_provenance

# Grab the actual module object for patching (the package __init__ re-exports
# the record_provenance FUNCTION, shadowing the submodule for dotted patching).
record_provenance_module = sys.modules["cognee.tasks.provenance.record_provenance"]
cognify_module = sys.modules["cognee.api.v1.cognify.cognify"]


class FakeEdge:
    def __init__(self, relationship_type):
        self.relationship_type = relationship_type


class FakeEntity:
    def __init__(self, name, entity_type_name=None, relations=None):
        self.id = uuid4()
        self.name = name
        self.is_a = SimpleNamespace(name=entity_type_name) if entity_type_name else None
        self.relations = relations or []


class FakeDocument:
    def __init__(self):
        self.id = uuid4()
        self.name = "doc.txt"
        self.raw_data_location = "/tmp/doc.txt"


class FakeChunk:
    def __init__(self, document, contains):
        self.id = uuid4()
        self.is_part_of = document
        self.contains = contains
        self.chunk_index = 3
        self.chunk_size = 42


class FakeSummary:
    def __init__(self, chunk):
        self.id = uuid4()
        self.made_from = chunk


def _pipeline_data():
    document = FakeDocument()
    target = FakeEntity("Acme", "Organization")
    entity = FakeEntity("Alice", "Person", relations=[(FakeEdge("works_at"), target)])
    chunk = FakeChunk(document, contains=[entity, (FakeEdge("mentions"), target)])
    return document, chunk, entity, target, [FakeSummary(chunk)]


def _ctx(dataset_id=None, pipeline_run_id=None):
    return PipelineContext(
        user=SimpleNamespace(email="user@example.com", id=uuid4()),
        data_item=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=dataset_id or uuid4()),
        pipeline_run_id=pipeline_run_id or uuid4(),
    )


class TestRecordProvenanceTask:
    @pytest.mark.asyncio
    async def test_writes_all_four_entry_types(self, manager):
        document, chunk, entity, target, data_points = _pipeline_data()
        ctx = _ctx()
        scope = str(ctx.dataset.id)

        result = await record_provenance(data_points, ctx=ctx)
        assert result is data_points

        entries = {entry.entity_id: entry for entry in await storage.retrieve_all()}
        run_id = str(ctx.pipeline_run_id)
        expected_ref = make_source_ref_key(ctx.dataset.id, ctx.data_item.id)

        # Every ledger key is namespaced by the dataset id (tenant isolation
        # over globally deterministic uuid5 entity ids).
        doc_entry = entries[f"{scope}:{document.id}"]
        assert doc_entry.entity_type == "document"
        assert doc_entry.source_document == "/tmp/doc.txt"
        assert doc_entry.parent_entity_id is None

        chunk_entry = entries[f"{scope}:{chunk.id}"]
        assert chunk_entry.entity_type == "chunk"
        assert chunk_entry.parent_entity_id == f"{scope}:{document.id}"
        assert chunk_entry.derived_from_id == f"{scope}:{document.id}"
        assert chunk_entry.start_index == chunk_entry.end_index == 3
        assert chunk_entry.metadata == {"chunk_size": 42}

        entity_entry = entries[f"{scope}:{entity.id}"]
        assert entity_entry.entity_type == "entity"
        assert entity_entry.metadata == {"name": "Alice", "type": "Person"}
        # Source-heuristic parent: the chunk is already tracked.
        assert entity_entry.parent_entity_id == f"{scope}:{chunk.id}"

        mention_rel = entries[f"rel:{scope}:{chunk.id}:mentions:{scope}:{target.id}"]
        works_rel = entries[f"rel:{scope}:{entity.id}:works_at:{scope}:{target.id}"]
        assert mention_rel.entity_type == works_rel.entity_type == "relationship"
        assert works_rel.used_entities == [f"{scope}:{entity.id}", f"{scope}:{target.id}"]
        assert works_rel.metadata == {"relationship_name": "works_at"}

        for entry in entries.values():
            assert entry.bundle_id == run_id
            assert entry.activity_id == f"cognify:{run_id}"
            assert entry.agent_id == "user@example.com"
            assert entry.source_ref_key == expected_ref

        stats = await manager.get_statistics()
        assert stats["entity_types"] == {
            "document": 1,
            "chunk": 1,
            "entity": 2,
            "relationship": 2,
        }
        assert (await manager.verify_chain())["valid"] is True
        assert (await manager.check())["valid"] is True

    @pytest.mark.asyncio
    async def test_single_chained_transaction_per_invocation(self, manager, monkeypatch):
        calls = []
        original = storage.append_chained_many

        async def counting(write_fns):
            calls.append(len(write_fns))
            return await original(write_fns)

        monkeypatch.setattr(storage, "append_chained_many", counting)
        _, _, _, _, data_points = _pipeline_data()
        await record_provenance(data_points, ctx=_ctx())
        # One batched append for the whole invocation, never one per entry.
        assert calls == [6]

    @pytest.mark.asyncio
    async def test_no_ctx_degrades_to_entries_without_source_ref(self, manager):
        _, chunk, _, _, data_points = _pipeline_data()
        result = await record_provenance(data_points, ctx=None)
        assert result is data_points

        entries = await storage.retrieve_all()
        assert entries, "entries must still be written without ctx"
        for entry in entries:
            assert entry.source_ref_key is None
            assert entry.bundle_id is None
            assert entry.activity_id == "cognify"
            assert entry.agent_id == "cognee"

    @pytest.mark.asyncio
    async def test_document_not_reversioned_across_batches_of_same_run(self, manager):
        # The task runs once per chunk batch; a document spanning batches must
        # be tracked once per run, not once per batch.
        document = FakeDocument()
        ctx = _ctx()
        scope = str(ctx.dataset.id)
        first_batch = [FakeSummary(FakeChunk(document, contains=[]))]
        second_batch = [FakeSummary(FakeChunk(document, contains=[]))]

        await record_provenance(first_batch, ctx=ctx)
        await record_provenance(second_batch, ctx=ctx)

        entries = await storage.retrieve_all()
        doc_key = f"{scope}:{document.id}"
        archives = [entry for entry in entries if entry.entity_id.startswith(f"{doc_key}:v:")]
        assert archives == []  # no forged version history
        history = await manager.revision_history(doc_key)
        assert len(history) == 1

        # A NEW run of the same dataset re-tracks (and versions) the document.
        await record_provenance(
            [FakeSummary(FakeChunk(document, contains=[]))],
            ctx=_ctx(dataset_id=ctx.dataset.id),
        )
        assert len(await manager.revision_history(doc_key)) == 2

    @pytest.mark.asyncio
    async def test_dataset_scoping_isolates_tenants(self, manager):
        # Same (deterministic) entity ids ingested under two datasets must not
        # share a version chain.
        document, chunk, entity, _, _ = _pipeline_data()
        data_points_a = [FakeSummary(chunk)]
        ctx_a, ctx_b = _ctx(), _ctx()

        await record_provenance(data_points_a, ctx=ctx_a)
        await record_provenance(data_points_a, ctx=ctx_b)

        key_a = f"{ctx_a.dataset.id}:{entity.id}"
        key_b = f"{ctx_b.dataset.id}:{entity.id}"
        entry_a = await manager.get_provenance(key_a)
        entry_b = await manager.get_provenance(key_b)
        assert entry_a is not None and entry_b is not None
        assert entry_a["source_ref_key"] != entry_b["source_ref_key"]
        # No cross-tenant archiving happened.
        assert entry_a["previous_version_id"] is None
        assert entry_b["previous_version_id"] is None

    @pytest.mark.asyncio
    async def test_custom_datapoint_schema_uses_generic_traversal(self, manager):
        # Custom graph_model DataPoints (no made_from/is_part_of/contains) must
        # still land entity + relationship rows, via get_graph_from_model.
        class CustomLeaf(DataPoint):
            name: str

        class CustomRoot(DataPoint):
            name: str
            points_to: CustomLeaf

        leaf = CustomLeaf(name="leaf")
        root = CustomRoot(name="root", points_to=leaf)

        await record_provenance([root], ctx=None)

        entries = {entry.entity_id: entry for entry in await storage.retrieve_all()}
        assert str(root.id) in entries
        assert str(leaf.id) in entries
        relationships = [entry for entry in entries.values() if entry.entity_type == "relationship"]
        assert len(relationships) == 1
        assert relationships[0].used_entities == [str(root.id), str(leaf.id)]
        assert (await manager.verify_chain())["valid"] is True

    @pytest.mark.asyncio
    async def test_generic_walk_without_the_input_attributes_to_nothing(self, manager):
        # get_graph_from_model may return a node set that does not include the item
        # it was given. Attributing those entries to the absent item would forge a
        # provenance chain pointing at something that was never written.
        class CustomTail(DataPoint):
            name: str

        class CustomLeaf(DataPoint):
            name: str
            points_to: CustomTail

        class CustomContainer(DataPoint):
            name: str
            holds: CustomLeaf
            metadata: dict = {"index_fields": [], "transparent": True}

        tail = CustomTail(name="tail")
        leaf = CustomLeaf(name="leaf", points_to=tail)
        container = CustomContainer(name="container", holds=leaf)

        result = await record_provenance([container], ctx=None)

        assert result == [container]
        entries = await storage.retrieve_all()
        entity_ids = {entry.entity_id for entry in entries}
        assert str(container.id) not in entity_ids
        # The leaf, the tail, and the one relationship between them.
        assert {str(leaf.id), str(tail.id)} <= entity_ids
        assert any(entry.entity_type == "relationship" for entry in entries)
        # Nothing may be attributed to an entity that was never written.
        assert all(entry.source_document == "" for entry in entries)
        assert all(entry.parent_entity_id is None for entry in entries)
        assert (await manager.verify_chain())["valid"] is True

    @pytest.mark.asyncio
    async def test_raw_item_without_document_still_tracked(self, manager):
        raw_item = SimpleNamespace(id=uuid4())  # no made_from, no is_part_of
        result = await record_provenance([raw_item], ctx=None)
        assert result == [raw_item]
        entries = await storage.retrieve_all()
        assert [entry.entity_id for entry in entries] == [str(raw_item.id)]
        assert entries[0].parent_entity_id is None

    @pytest.mark.asyncio
    async def test_manager_failure_never_raises(self, monkeypatch):
        def explode():
            raise RuntimeError("ledger down")

        monkeypatch.setattr(record_provenance_module, "get_provenance_manager", explode)
        _, _, _, _, data_points = _pipeline_data()
        result = await record_provenance(data_points, ctx=_ctx())
        assert result is data_points

    @pytest.mark.asyncio
    async def test_empty_input_is_a_noop(self):
        assert await record_provenance([]) == []


async def _task_names(provenance_flag, contradiction_flag=False):
    config = CognifyConfig(
        provenance_tracking=provenance_flag,
        contradiction_detection=contradiction_flag,
    )
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
    async def test_flag_off_pipeline_unchanged(self):
        names = await _task_names(provenance_flag=False)
        assert "record_provenance" not in names

    @pytest.mark.asyncio
    async def test_flag_on_splices_after_add_data_points_before_contradictions(self):
        names = await _task_names(provenance_flag=True, contradiction_flag=True)
        assert names.index("record_provenance") == names.index("add_data_points") + 1
        assert names.index("record_provenance") < names.index("detect_contradictions")
