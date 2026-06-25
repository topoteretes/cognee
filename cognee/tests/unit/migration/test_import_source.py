"""Unit tests for import_memory_source and the remember() MemorySource dispatch.

All tests are pure: no databases, no LLM calls, no network. The three sinks the
import orchestration can hit (run_custom_pipeline, add, the nested remember)
are monkeypatched so each test asserts exactly which of them fire per mode.
"""

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator

import pytest

from cognee.api.v1.remember.remember import RememberResult, remember
from cognee.modules.migration.cogx import COGXDocument, COGXEntity, COGXFact, COGXRecord
from cognee.modules.migration.import_source import import_memory_source
from cognee.modules.migration.sources import GoogleMemorySource
from cognee.modules.migration.sources.base import MemorySource

GOOGLE_ADK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "google_adk_export.json"

# Module handles for monkeypatching the seams import_memory_source resolves
# lazily at call time. importlib is required: plain ``import a.b.c as x``
# fails here because package __init__ files rebind submodule names to
# same-named functions (e.g. ``cognee.api.v1.remember.remember``).
add_module = importlib.import_module("cognee.api.v1.add")
pipeline_module = importlib.import_module("cognee.modules.run_custom_pipeline")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state = importlib.import_module("cognee.api.v1.serve.state")
shared_utils = importlib.import_module("cognee.shared.utils")
storage_module = importlib.import_module("cognee.tasks.storage.add_data_points")

UNRESOLVABLE_UUID = "0c113fd0-1234-4321-aaaa-bbbbccccdddd"


class FakeSource(MemorySource):
    """A MemorySource that streams a fixed list of records."""

    source_system = "fake"

    def __init__(self, records, mode="re-derive"):
        super().__init__(mode)
        self._records = records

    async def records(self) -> AsyncIterator[COGXRecord]:
        for record in self._records:
            yield record


def _sample_records():
    """One document, two entities, one fact between them.

    Entities carry no description and no entity_type, so re-derive mode emits
    no entity digest and preserve mode creates no EntityType nodes — keeping
    the expected counts below easy to reason about.
    """
    return [
        COGXDocument(external_system="fake", external_id="d1", content="hello", title="Doc"),
        COGXEntity(external_system="fake", external_id="n1", name="Alice"),
        COGXEntity(external_system="fake", external_id="n2", name="Bob"),
        COGXFact(
            external_system="fake",
            external_id="f1",
            subject_ref="n1",
            predicate="knows",
            object_ref="n2",
        ),
    ]


def install_sinks(monkeypatch):
    """Replace the orchestration seams with recorders.

    ``run_custom_pipeline`` simulates blocking execution by running each
    task's executable over the pipeline data, so the streaming import task
    actually streams — its storage flushes land in ``graph_flushes`` via the
    ``add_data_points`` sink. It returns a run-info dict so pipeline_run_id
    propagation can be asserted.
    """
    sinks = SimpleNamespace(pipeline_calls=[], add_calls=[], remember_calls=[], graph_flushes=[])

    async def fake_run_custom_pipeline(**kwargs):
        sinks.pipeline_calls.append(kwargs)
        for task in kwargs.get("tasks", []):
            executable = getattr(task, "executable", None)
            if executable is not None:
                await executable(kwargs.get("data"))
        return {"ds": SimpleNamespace(pipeline_run_id="run-123")}

    async def fake_add_data_points(data_points, custom_edges=None, ctx=None):
        sinks.graph_flushes.append({"nodes": list(data_points), "edges": list(custom_edges or [])})
        return data_points

    async def fake_add(data, **kwargs):
        sinks.add_calls.append({"data": data, **kwargs})

    async def fake_remember(data, dataset_name="main_dataset", **kwargs):
        sinks.remember_calls.append({"data": data, "dataset_name": dataset_name, **kwargs})
        result = RememberResult(status="completed", dataset_name=dataset_name)
        result.items_processed = len(data)
        return result

    monkeypatch.setattr(pipeline_module, "run_custom_pipeline", fake_run_custom_pipeline)
    monkeypatch.setattr(storage_module, "add_data_points", fake_add_data_points)
    monkeypatch.setattr(add_module, "add", fake_add)
    monkeypatch.setattr(remember_module, "remember", fake_remember)
    return sinks


def _flushed_nodes(sinks):
    return [node for flush in sinks.graph_flushes for node in flush["nodes"]]


def _flushed_edges(sinks):
    return [edge for flush in sinks.graph_flushes for edge in flush["edges"]]


def _summary_items(result):
    return [item for item in result.items if item.get("kind") == "migration_import"]


def _google_sample_entries():
    return [
        {
            "id": "memory-001",
            "content": {"parts": [{"text": "Alice works at Acme"}]},
            "timestamp": "2024-01-01T00:00:00Z",
            "custom_metadata": {
                "user_id": "u1",
                "app_name": "career_assistant",
                "session_id": "session_abc",
            },
        },
        {
            "id": "memory-002",
            "content": {"parts": [{"text": "User prefers tea"}]},
        },
    ]


def _google_empty_entries():
    return [
        {"id": "empty"},
        {"content": {"parts": [{"text": "   "}]}},
        {"content": {"parts": [{"inline_data": {"data": "abc"}}]}},
    ]


class TestGoogleMemoryImport:
    """Integration tests for GoogleMemorySource through import_memory_source."""

    def test_preserve_imports_fixture(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(GOOGLE_ADK_FIXTURE, mode="preserve")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(sinks.add_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.remember_calls == []
        assert sinks.graph_flushes == []

        add_call = sinks.add_calls[0]
        assert add_call["dataset_name"] == "ds"
        assert add_call["node_set"] == ["import:google_adk"]
        assert len(add_call["data"]) == 2
        assert add_call["data"][0].data == "Alice works at Acme"
        assert add_call["data"][1].data == "from file"

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "google_adk"
        assert summary["mode"] == "preserve"
        assert summary["record_counts"] == {"memory": 2}
        assert summary["graph_nodes"] == 0
        assert summary["graph_edges"] == 0
        assert summary["skipped_facts"] == 0
        assert summary["pipeline_run_id"] is None
        assert result.status == "completed"
        assert result.items_processed == 2
        assert result.dataset_name == "ds"
        assert result.elapsed_seconds is not None

    def test_preserve_data_items_carry_google_metadata(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="preserve")

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        items = sinks.add_calls[0]["data"]
        assert items[0].external_metadata["external_system"] == "google_adk"
        assert items[0].external_metadata["external_id"] == "memory-001"
        assert items[0].external_metadata["scope"] == {
            "user_id": "u1",
            "agent_id": "career_assistant",
            "session_id": "session_abc",
        }
        assert items[0].external_metadata["user_id"] == "u1"
        assert items[0].external_metadata["app_name"] == "career_assistant"

    def test_preserve_custom_node_set(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="preserve")

        asyncio.run(import_memory_source(source, dataset_name="ds", node_set=["custom-google"]))

        assert sinks.add_calls[0]["node_set"] == ["custom-google"]

    def test_preserve_memories_wrapper_shape(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(
            {"memories": [{"content": {"parts": [{"text": "wrapped memory"}]}}]},
            mode="preserve",
        )

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        (summary,) = _summary_items(result)
        assert summary["record_counts"] == {"memory": 1}
        assert sinks.add_calls[0]["data"][0].data == "wrapped memory"

    def test_preserve_skipped_entries_produce_no_data_items(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_empty_entries(), mode="preserve")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert sinks.add_calls == []
        (summary,) = _summary_items(result)
        assert summary["record_counts"] == {}
        assert result.items_processed == 0

    def test_re_derive_fires_nested_remember_only(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="re-derive")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(sinks.remember_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.add_calls == []
        assert sinks.graph_flushes == []

        call = sinks.remember_calls[0]
        assert call["dataset_name"] == "ds"
        assert call["node_set"] == ["import:google_adk"]
        assert len(call["data"]) == 2
        assert call["data"][0].data == "Alice works at Acme"
        assert call["data"][1].data == "User prefers tea"

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "google_adk"
        assert summary["mode"] == "re-derive"
        assert summary["record_counts"] == {"memory": 2}
        assert summary["graph_nodes"] == 0
        assert summary["graph_edges"] == 0
        assert result.items_processed == 2

    def test_hybrid_memory_only_fires_nested_remember(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="hybrid")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(sinks.remember_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.add_calls == []

        (summary,) = _summary_items(result)
        assert summary["mode"] == "hybrid"
        assert summary["graph_nodes"] == 0
        assert summary["graph_edges"] == 0
        assert result.items_processed == 2

    def test_preserve_non_replayable_uses_buffered_add(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="preserve")
        source.replayable = False

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(sinks.add_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.remember_calls == []
        assert len(sinks.add_calls[0]["data"]) == 2
        (summary,) = _summary_items(result)
        assert summary["mode"] == "preserve"
        assert summary["record_counts"] == {"memory": 2}

    def test_remember_dispatch_preserve(self, monkeypatch):
        monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
        monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(GOOGLE_ADK_FIXTURE, mode="preserve")

        result = asyncio.run(remember(source, "ds"))

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "google_adk"
        assert summary["mode"] == "preserve"
        assert summary["record_counts"] == {"memory": 2}
        assert len(sinks.add_calls) == 1
        assert sinks.remember_calls == []

    def test_remember_dispatch_re_derive(self, monkeypatch):
        monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
        monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
        sinks = install_sinks(monkeypatch)
        source = GoogleMemorySource(_google_sample_entries(), mode="re-derive")

        result = asyncio.run(remember(source, "ds"))

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "google_adk"
        assert summary["mode"] == "re-derive"
        assert len(sinks.remember_calls) == 1
        assert sinks.add_calls == []


class TestImportMemorySourceModes:
    def test_re_derive_fires_only_nested_remember(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="re-derive")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        # Re-derive: everything goes through the nested remember (add+cognify);
        # no direct graph storage, no raw add.
        assert len(sinks.remember_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.add_calls == []

        call = sinks.remember_calls[0]
        assert call["dataset_name"] == "ds"
        assert call["node_set"] == ["import:fake"]
        assert call["run_in_background"] is False
        # 1 document + 1 facts digest (entities have no description, so no
        # entity digest); the source graph becomes text, not graph batches.
        labels = [item.label for item in call["data"]]
        assert len(call["data"]) == 2
        assert "Imported facts" in labels

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "fake"
        assert summary["mode"] == "re-derive"
        assert summary["record_counts"] == {"document": 1, "entity": 2, "fact": 1}
        assert summary["graph_nodes"] == 0
        assert summary["graph_edges"] == 0
        assert summary["skipped_facts"] == 0
        # Nested remember processed 2 data items; no graph nodes to add.
        assert result.items_processed == 2

    def test_preserve_streams_graph_and_chunked_add(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        # Preserve + replayable: the graph imports through ONE streaming
        # pipeline task, raw content through chunked add (no cognify), and
        # the nested remember never fires.
        assert len(sinks.pipeline_calls) == 1
        assert len(sinks.add_calls) == 1
        assert sinks.remember_calls == []

        pipeline = sinks.pipeline_calls[0]
        assert pipeline["dataset"] == "ds"
        assert pipeline["pipeline_name"] == "migration_import_pipeline"
        assert pipeline["run_in_background"] is False
        # The pipeline data is a single stream handle, not materialized batches.
        assert len(pipeline["data"]) == 1
        assert pipeline["data"][0].data["kind"] == "graph_stream"

        # The streamed flushes carry the actual graph content.
        nodes = _flushed_nodes(sinks)
        edges = _flushed_edges(sinks)
        assert sorted(node.name for node in nodes) == ["Alice", "Bob"]
        assert len(edges) == 1
        assert edges[0][2] == "knows"

        add_call = sinks.add_calls[0]
        assert add_call["dataset_name"] == "ds"
        assert add_call["node_set"] == ["import:fake"]
        assert len(add_call["data"]) == 1  # the document, stored raw

        (summary,) = _summary_items(result)
        assert summary["mode"] == "preserve"
        assert summary["graph_nodes"] == 2
        assert summary["graph_edges"] == 1
        assert summary["skipped_facts"] == 0
        assert summary["pipeline_run_id"] == "run-123"
        # 1 raw data item + 2 graph nodes.
        assert result.items_processed == 3
        assert result.status == "completed"
        assert result.pipeline_run_id == "run-123"
        assert result.dataset_name == "ds"
        assert result.elapsed_seconds is not None

    def test_preserve_streaming_flushes_in_bounded_batches(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        loader_module = importlib.import_module("cognee.modules.migration.loader")
        monkeypatch.setattr(loader_module, "BATCH_NODE_TARGET", 3)

        records = [
            COGXEntity(external_system="fake", external_id=f"n{i}", name=f"Entity{i}")
            for i in range(8)
        ]
        source = FakeSource(records, mode="preserve")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        # 8 entities with a target of 3 → at least 3 node flushes, none over
        # the bound: memory stays bounded by one batch, not the whole graph.
        node_flushes = [flush for flush in sinks.graph_flushes if flush["nodes"]]
        assert len(node_flushes) >= 3
        assert all(len(flush["nodes"]) <= 3 for flush in node_flushes)
        assert len(_flushed_nodes(sinks)) == 8
        (summary,) = _summary_items(result)
        assert summary["graph_nodes"] == 8

    def test_preserve_non_replayable_uses_buffered_batches(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")
        source.replayable = False

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        # One-shot sources cannot be re-streamed: the buffered path wraps
        # bounded graph batches as pipeline data instead.
        pipeline = sinks.pipeline_calls[0]
        batches = [item.data for item in pipeline["data"]]
        assert sum(len(batch["nodes"]) for batch in batches) == 2
        assert sum(len(batch["edges"]) for batch in batches) == 1

        (summary,) = _summary_items(result)
        assert summary["graph_nodes"] == 2
        assert summary["graph_edges"] == 1
        assert summary["pipeline_run_id"] == "run-123"
        assert result.pipeline_run_id == "run-123"

    def test_preserve_background_reports_started(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")

        result = asyncio.run(
            import_memory_source(source, dataset_name="ds", run_in_background=True)
        )

        assert sinks.pipeline_calls[0]["run_in_background"] is True
        assert result.status == "started"
        assert result.pipeline_run_id == "run-123"
        (summary,) = _summary_items(result)
        assert summary["graph_import"] == "running"

    def test_hybrid_fires_pipeline_and_nested_remember(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="hybrid")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        # Hybrid: graph batches are preserved AND raw content is cognified.
        assert len(sinks.pipeline_calls) == 1
        assert len(sinks.remember_calls) == 1
        assert sinks.add_calls == []

        call = sinks.remember_calls[0]
        assert call["node_set"] == ["import:fake"]
        assert len(call["data"]) == 1  # the document only; the graph is preserved

        (summary,) = _summary_items(result)
        assert summary["graph_nodes"] == 2
        assert summary["graph_edges"] == 1
        # Nested remember processed 1 data item, plus 2 preserved graph nodes.
        assert result.items_processed == 3

    def test_run_in_background_threads_into_pipeline(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="hybrid")

        asyncio.run(import_memory_source(source, dataset_name="ds", run_in_background=True))

        assert sinks.pipeline_calls[0]["run_in_background"] is True
        assert sinks.remember_calls[0]["run_in_background"] is True

    def test_explicit_node_set_overrides_default(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")

        asyncio.run(import_memory_source(source, dataset_name="ds", node_set=["custom"]))

        assert sinks.add_calls[0]["node_set"] == ["custom"]


class TestSkippedFacts:
    def test_unresolvable_uuid_fact_is_skipped_and_surfaced(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(
            [
                COGXEntity(external_system="fake", external_id="n1", name="Alice"),
                COGXFact(
                    external_system="fake",
                    external_id="f1",
                    subject_ref=UNRESOLVABLE_UUID,
                    predicate="knows",
                    object_ref="n1",
                ),
            ],
            mode="preserve",
        )

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        (summary,) = _summary_items(result)
        assert summary["skipped_facts"] == 1
        assert summary["graph_edges"] == 0
        assert summary["graph_nodes"] == 1

        # The skipped fact must never fabricate a UUID-named entity.
        assert [node.name for node in _flushed_nodes(sinks)] == ["Alice"]
        assert _flushed_edges(sinks) == []


class TestRememberDispatch:
    """remember() routing for MemorySource inputs (guards, kwargs forwarding)."""

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
        monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)

    def test_remote_client_guard_raises_with_push_hint(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        monkeypatch.setattr(serve_state, "get_remote_client", lambda: object())
        source = FakeSource(_sample_records(), mode="preserve")

        with pytest.raises(ValueError, match=r"cognee\.push\(\)"):
            asyncio.run(remember(source))

        # The guard fires before any local work happens.
        assert sinks.pipeline_calls == []
        assert sinks.add_calls == []
        assert sinks.remember_calls == []

    def test_session_id_raises(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="re-derive")

        with pytest.raises(ValueError, match="session_id"):
            asyncio.run(remember(source, session_id="session-1"))

        assert sinks.remember_calls == []

    def test_kwargs_forwarded_to_nested_remember(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="re-derive")

        result = asyncio.run(
            remember(
                source,
                "ds",
                custom_prompt="extract people",
                chunk_size=512,
                self_improvement=False,
            )
        )

        call = sinks.remember_calls[0]
        assert call["custom_prompt"] == "extract people"
        assert call["chunk_size"] == 512
        assert call["self_improvement"] is False
        assert call["chunker"] is None
        assert call["dataset_name"] == "ds"
        assert call["node_set"] == ["import:fake"]
        assert _summary_items(result)

    def test_dispatch_returns_import_result(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")

        result = asyncio.run(remember(source, "ds"))

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "fake"
        assert summary["mode"] == "preserve"
        assert len(sinks.pipeline_calls) == 1
        assert len(sinks.add_calls) == 1
        assert sinks.remember_calls == []
