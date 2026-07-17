"""Unit tests for import_memory_source and the remember() MemorySource dispatch.

All tests are pure: no databases, no LLM calls, no network. The three sinks the
import orchestration can hit (run_custom_pipeline, add, the nested remember)
are monkeypatched so each test asserts exactly which of them fire per mode.
"""

import asyncio
import importlib
from types import SimpleNamespace
from typing import AsyncIterator
from uuid import uuid4

import pytest

from cognee.api.v1.remember.remember import RememberResult, remember
from cognee.modules.migration.cogx import COGXDocument, COGXEntity, COGXFact, COGXRecord
from cognee.modules.migration.import_source import import_memory_source
from cognee.modules.migration.sources.base import MemorySource

# Module handles for monkeypatching the seams import_memory_source resolves
# lazily at call time. importlib is required: plain ``import a.b.c as x``
# fails here because package __init__ files rebind submodule names to
# same-named functions (e.g. ``cognee.api.v1.remember.remember``).
add_module = importlib.import_module("cognee.api.v1.add")
migrations_startup_module = importlib.import_module("cognee.modules.migrations.startup")
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
    sinks = SimpleNamespace(
        pipeline_calls=[],
        add_calls=[],
        remember_calls=[],
        graph_flushes=[],
        migration_gate_calls=[],
    )

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

    async def fake_run_migrations_and_block(datasets, user):
        # writes_before_gate lets tests assert the gate fired before any
        # imported data was stored (stamping must precede the rows).
        sinks.migration_gate_calls.append(
            {
                "datasets": datasets,
                "user": user,
                "writes_before_gate": len(sinks.add_calls) + len(sinks.graph_flushes),
            }
        )

    monkeypatch.setattr(pipeline_module, "run_custom_pipeline", fake_run_custom_pipeline)
    monkeypatch.setattr(storage_module, "add_data_points", fake_add_data_points)
    monkeypatch.setattr(add_module, "add", fake_add)
    monkeypatch.setattr(remember_module, "remember", fake_remember)
    monkeypatch.setattr(
        migrations_startup_module, "run_migrations_and_block", fake_run_migrations_and_block
    )
    return sinks


def _flushed_nodes(sinks):
    return [node for flush in sinks.graph_flushes for node in flush["nodes"]]


def _flushed_edges(sinks):
    return [edge for flush in sinks.graph_flushes for edge in flush["edges"]]


def _summary_items(result):
    return [item for item in result.items if item.get("kind") == "migration_import"]


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


class TestMigrationGate:
    """import_memory_source must take the run_migrations_and_block gate.

    The gate stamps a fresh store's migration revision at head BEFORE the
    imported rows arrive; skipping it leaves the populated store with no
    recorded revision, so the first migration-aware startup replays the whole
    data-migration chain over the freshly imported data.
    """

    @pytest.mark.parametrize("mode", ["preserve", "re-derive", "hybrid"])
    def test_gate_fires_once_before_any_write(self, monkeypatch, mode):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode=mode)

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        (gate_call,) = sinks.migration_gate_calls
        assert gate_call["datasets"] == "ds"
        assert gate_call["writes_before_gate"] == 0


class TestNodeIdScheme:
    """Imported entities must carry cognee's CURRENT class-namespaced ids
    (Entity.id_for / EntityType.id_for — the identity_fields scheme cognify
    uses), not the legacy bare generate_node_id hash. Stale-scheme ids form a
    disconnected parallel vocabulary and are invisible to the namespace
    migrations, which the import stamps as already applied."""

    def test_imported_entity_ids_use_cognee_identity_scheme(self, monkeypatch):
        from cognee.modules.engine.models import Entity

        sinks = install_sinks(monkeypatch)
        source = FakeSource(_sample_records(), mode="preserve")

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        alice = next(n for n in _flushed_nodes(sinks) if getattr(n, "name", None) == "Alice")
        assert alice.id == Entity.id_for("Alice")

    def _cognee_archive_records(self):
        a1, a2, b = str(uuid4()), str(uuid4()), str(uuid4())
        records = [
            COGXEntity(external_system="cognee", external_id=a1, name="Alice"),
            COGXEntity(external_system="cognee", external_id=a2, name="Alice"),
            COGXEntity(external_system="cognee", external_id=b, name="Bob"),
            COGXFact(
                external_system="cognee",
                external_id="f1",
                subject_ref=a1,
                predicate="knows",
                object_ref=b,
            ),
            COGXFact(
                external_system="cognee",
                external_id="f2",
                subject_ref=a2,
                predicate="knows",
                object_ref=b,
            ),
        ]
        return records, {a1, a2}

    @pytest.mark.parametrize("replayable", [True, False])
    def test_cognee_archives_preserve_source_uuids(self, monkeypatch, replayable):
        """Cognee-to-cognee transfers are exact copies: entity nodes keep the
        source UUIDs verbatim, so same-named-but-distinct source entities stay
        distinct and their edges never collide (no merge, nothing to dedupe)."""
        sinks = install_sinks(monkeypatch)
        records, alice_ids = self._cognee_archive_records()
        source = FakeSource(records, mode="preserve")
        source.source_system = "cognee"
        source.replayable = replayable

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        alices = [n for n in _flushed_nodes(sinks) if getattr(n, "name", None) == "Alice"]
        assert {str(node.id) for node in alices} == alice_ids
        assert len(_flushed_edges(sinks)) == 2


class TestResolvedEdgeDeduplication:
    """Facts with distinct external refs can RESOLVE to the same
    (source, target, relationship) edge key — entities merge by name. Such
    duplicates must be dropped at translation (first fact wins): re-MERGEing
    an existing edge fires a rel-property update, which crashes Ladybug's
    committed-in-memory row lookup during fresh bulk imports."""

    def _records_with_resolved_duplicate(self):
        # a1 and a2 are distinct records but merge into ONE node (same name),
        # so both facts resolve to the same edge key (Alice -knows-> Bob).
        return [
            COGXEntity(external_system="fake", external_id="a1", name="Alice"),
            COGXEntity(external_system="fake", external_id="a2", name="Alice"),
            COGXEntity(external_system="fake", external_id="b1", name="Bob"),
            COGXFact(
                external_system="fake",
                external_id="f1",
                subject_ref="a1",
                predicate="knows",
                object_ref="b1",
            ),
            COGXFact(
                external_system="fake",
                external_id="f2",
                subject_ref="a2",
                predicate="knows",
                object_ref="b1",
            ),
        ]

    def test_streaming_import_dedupes_resolved_edges(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(self._records_with_resolved_duplicate(), mode="preserve")

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(_flushed_edges(sinks)) == 1
        (summary,) = _summary_items(result)
        assert summary["graph_edges"] == 1
        assert summary["deduped_edges"] == 1

    def test_buffered_import_dedupes_resolved_edges(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        source = FakeSource(self._records_with_resolved_duplicate(), mode="preserve")
        source.replayable = False  # force the buffered path

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(_flushed_edges(sinks)) == 1


class TestSocialLayerRestore:
    """Archives exported with include_permissions carry the social layer:
    the import must run AS the archived owner (per-dataset databases derive
    their physical location from the owner id, so ownership cannot be
    reassigned after the rows land) and re-apply the grants afterwards.
    Sources without a social layer keep today's behavior untouched."""

    def _social_source(self):
        source = FakeSource(_sample_records(), mode="preserve")
        source.source_system = "cognee"
        source.social_layer = {
            "owner": {"email": "owner@example.com", "hashed_password": "h", "is_active": True},
            "grants": [
                {
                    "user": {"email": "reviewer@example.com", "hashed_password": "h2"},
                    "permissions": ["read"],
                }
            ],
        }
        return source

    def test_import_runs_as_archived_owner_and_applies_grants(self, monkeypatch):
        import cognee.modules.migration.import_source as import_source_module

        sinks = install_sinks(monkeypatch)
        owner_stub = SimpleNamespace(id=uuid4(), email="owner@example.com")
        superuser = SimpleNamespace(id=uuid4(), email="admin@example.com", is_superuser=True)
        ensured, grant_calls = [], []

        async def fake_ensure_user(payload):
            ensured.append(payload["email"])
            return owner_stub

        async def fake_apply_grants(source, dataset_name, owner, importer):
            grant_calls.append({"dataset_name": dataset_name, "owner": owner})

        monkeypatch.setattr(import_source_module, "_ensure_user", fake_ensure_user)
        monkeypatch.setattr(import_source_module, "_apply_social_grants", fake_apply_grants)
        source = self._social_source()

        asyncio.run(import_memory_source(source, dataset_name="ds", user=superuser))

        assert ensured == ["owner@example.com"]
        # Every write ran AS the owner, not the calling user.
        assert all(call.get("user") is owner_stub for call in sinks.add_calls)
        assert all(call.get("user") is owner_stub for call in sinks.pipeline_calls)
        (grant_call,) = grant_calls
        assert grant_call["owner"] is owner_stub
        assert grant_call["dataset_name"] == "ds"

    def test_social_layer_requires_superuser_importer(self, monkeypatch):
        """A crafted archive could otherwise mint arbitrary accounts —
        including superusers — with attacker-chosen credentials, via the SDK
        or the /v1/remember archive-upload endpoint."""
        from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

        install_sinks(monkeypatch)
        regular = SimpleNamespace(id=uuid4(), email="user@example.com", is_superuser=False)
        source = self._social_source()

        with pytest.raises(PermissionDeniedError, match="superuser"):
            asyncio.run(import_memory_source(source, dataset_name="ds", user=regular))

    def test_source_without_social_layer_keeps_calling_user(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        caller = SimpleNamespace(id=uuid4(), email="caller@example.com")
        source = FakeSource(_sample_records(), mode="preserve")
        assert source.social_layer is None

        asyncio.run(import_memory_source(source, dataset_name="ds", user=caller))

        assert all(call.get("user") is caller for call in sinks.add_calls)
        assert all(call.get("user") is caller for call in sinks.pipeline_calls)


class TestSourceRevisionRestamp:
    """The import aligns the target's migration stamp with a cognee-origin
    archive's source revision (see _restamp_to_source_revision): backward
    only, known revisions only, so the next migration run replays exactly
    archive -> head over the imported rows."""

    CHAIN = ["rev_a", "rev_b", "rev_c"]

    def test_stamps_backward_when_archive_is_behind(self):
        from cognee.modules.migration.import_source import _revision_to_stamp

        assert _revision_to_stamp("rev_a", "rev_c", self.CHAIN) == "rev_a"

    def test_never_stamps_forward_or_same(self):
        from cognee.modules.migration.import_source import _revision_to_stamp

        assert _revision_to_stamp("rev_c", "rev_a", self.CHAIN) is None
        assert _revision_to_stamp("rev_b", "rev_b", self.CHAIN) is None

    def test_unknown_or_missing_revisions_leave_stamp_untouched(self):
        from cognee.modules.migration.import_source import _revision_to_stamp

        assert _revision_to_stamp(None, "rev_c", self.CHAIN) is None
        assert _revision_to_stamp("newer_rev", "rev_c", self.CHAIN) is None
        assert _revision_to_stamp("rev_a", "newer_rev", self.CHAIN) is None
        # Unstamped store (base) is already minimal — nothing to lower.
        assert _revision_to_stamp("rev_a", None, self.CHAIN) is None

    def test_import_runs_restamp_after_the_rows_land(self, monkeypatch):
        import cognee.modules.migration.import_source as import_source_module

        sinks = install_sinks(monkeypatch)
        restamp_calls = []

        async def fake_restamp(source, dataset_name, user):
            restamp_calls.append(
                {
                    "source": source,
                    "dataset_name": dataset_name,
                    "flushes_before_restamp": len(sinks.graph_flushes),
                }
            )

        monkeypatch.setattr(import_source_module, "_restamp_to_source_revision", fake_restamp)
        source = FakeSource(_sample_records(), mode="preserve")

        asyncio.run(import_memory_source(source, dataset_name="ds"))

        (call,) = restamp_calls
        assert call["source"] is source
        assert call["dataset_name"] == "ds"
        # The restamp must run AFTER the import wrote its rows.
        assert call["flushes_before_restamp"] > 0

    def test_source_without_revision_skips_restamp_without_db_access(self):
        """External sources (Mem0/Zep/Letta) carry no revision: the helper
        must return before touching migrations or the database."""
        from cognee.modules.migration.import_source import _restamp_to_source_revision

        source = FakeSource(_sample_records(), mode="preserve")
        assert source.migration_revision is None
        # Would raise on any DB/engine access — pure short-circuit expected.
        asyncio.run(_restamp_to_source_revision(source, "ds", None))


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


class TestLangMemImport:
    def test_preserve_imports_semantic_memories(self, monkeypatch):
        from cognee.modules.migration.sources.langmem import LangMemSource

        sinks = install_sinks(monkeypatch)
        source = LangMemSource(
            [
                {
                    "namespace": ["user-42", "memories"],
                    "key": "mem-001",
                    "value": {
                        "kind": "Memory",
                        "content": {"content": "User prefers dark mode"},
                    },
                }
            ],
            mode="preserve",
        )

        result = asyncio.run(import_memory_source(source, dataset_name="ds"))

        assert len(sinks.add_calls) == 1
        assert sinks.pipeline_calls == []
        assert sinks.remember_calls == []
        assert len(sinks.add_calls[0]["data"]) == 1
        assert sinks.add_calls[0]["node_set"] == ["import:langmem"]

        (summary,) = _summary_items(result)
        assert summary["source_system"] == "langmem"
        assert summary["mode"] == "preserve"
        assert summary["record_counts"] == {"memory": 1}
        assert result.status == "completed"
