"""The ``vector_compaction`` job: its gate, its stats mapping, and a real pass."""

import importlib
from uuid import uuid4

import pytest
from pydantic import BaseModel

from cognee.infrastructure.databases.vector.config import get_vectordb_config
from cognee.modules.maintenance import (
    REASON_BACKEND_UNSUPPORTED,
    REASON_DISABLED_BY_CONFIG,
    MaintenanceContext,
)
from cognee.modules.maintenance.jobs import VectorCompactionJob
from cognee.modules.maintenance.jobs import vector_compaction as job_module
from cognee.modules.maintenance.jobs.vector_compaction import result_from_stats

CTX = MaintenanceContext(
    pipeline_name="cognify_pipeline", pipeline_run_id=None, dataset=None, user=None
)


@pytest.fixture
def vector_settings(monkeypatch):
    def configure(**settings):
        for key, value in settings.items():
            monkeypatch.setenv(key.upper(), str(value))
        get_vectordb_config.cache_clear()

    get_vectordb_config.cache_clear()
    yield configure
    get_vectordb_config.cache_clear()


def _no_engine(monkeypatch):
    async def must_not_be_called():
        raise AssertionError("the gate created the vector engine")

    monkeypatch.setattr(job_module, "get_vector_engine_async", must_not_be_called)


def test_gate_skips_when_compaction_is_disabled(vector_settings, monkeypatch):
    vector_settings(vector_db_compaction_enabled="false")
    _no_engine(monkeypatch)
    assert VectorCompactionJob().gate(CTX) == REASON_DISABLED_BY_CONFIG


@pytest.mark.parametrize("provider", ["pgvector", "turso", "neptune_analytics"])
def test_gate_skips_providers_that_do_not_compact(vector_settings, monkeypatch, provider):
    vector_settings(vector_db_compaction_enabled="true")
    monkeypatch.setattr(
        job_module, "get_vectordb_context_config", lambda: {"vector_db_provider": provider}
    )
    _no_engine(monkeypatch)
    assert VectorCompactionJob().gate(CTX) == REASON_BACKEND_UNSUPPORTED


def test_gate_opens_for_lancedb(vector_settings, monkeypatch):
    vector_settings(vector_db_compaction_enabled="true")
    monkeypatch.setattr(
        job_module, "get_vectordb_context_config", lambda: {"vector_db_provider": "LanceDB"}
    )
    assert VectorCompactionJob().gate(CTX) is None


def test_stats_mapping():
    worked = result_from_stats(
        "vector_compaction",
        {
            "A": {
                "planned_tasks": 3,
                "executed_tasks": 2,
                "fragments_removed": 6,
                "fragments_added": 2,
                "old_versions_removed": 5,
                "versions_pending": 1,
            },
            "B": {"error": "boom"},
        },
    )
    assert worked.status == "completed"
    assert worked.counts == {
        "collections": 2,
        "tasks_executed": 2,
        "tasks_pending": 1,
        "fragments_removed": 6,
        "fragments_added": 2,
        "versions_removed": 5,
        "versions_pending": 1,
        "collection_errors": 1,
        "prune_errors": 0,
    }

    idle = result_from_stats("vector_compaction", {"A": {"planned_tasks": 0, "executed_tasks": 0}})
    assert idle.status == "already_completed"

    for reason in ("remote_store", "in_progress", "lance_core_mismatch", "disabled"):
        skipped = result_from_stats("vector_compaction", {"skipped": reason})
        assert (skipped.status, skipped.reason) == ("skipped", reason)

    assert result_from_stats("vector_compaction", {}).status == "already_completed"
    assert result_from_stats("vector_compaction", None).status == "already_completed"


@pytest.mark.asyncio
async def test_run_uses_the_async_engine_getter(monkeypatch):
    """The deprecated sync getter warns on every run and fails under -W error."""
    import warnings

    class Engine:
        async def compact(self):
            return {"A": {"planned_tasks": 1, "executed_tasks": 1}}

    async def getter():
        return Engine()

    assert not hasattr(job_module, "get_vector_engine")
    monkeypatch.setattr(job_module, "get_vector_engine_async", getter)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = await VectorCompactionJob().run(CTX)
    assert result.status == "completed"


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _Payload(BaseModel):
    slot: int


@pytest.mark.asyncio
@pytest.mark.parametrize("subprocess_enabled", [False, True], ids=["local", "subprocess"])
async def test_a_real_pass_through_the_runner(
    tmp_path, monkeypatch, vector_settings, subprocess_enabled
):
    """End to end: the runner's job compacts a real LanceDB store, in both modes."""
    pytest.importorskip("lancedb")
    from cognee.infrastructure.databases.vector.config import VectorConfig
    from cognee.modules.maintenance import run_maintenance

    vector_settings(vector_db_compaction_enabled="true", vector_db_compaction_retention_seconds=0)
    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    config = VectorConfig(
        vector_db_provider="lancedb",
        vector_db_url=str(tmp_path / "db"),
        vector_db_name=f"maintenance_{subprocess_enabled}",
        vector_db_subprocess_enabled=subprocess_enabled,
    )
    adapter = factory.create_vector_engine(**config.to_dict())

    async def getter():
        return adapter

    monkeypatch.setattr(job_module, "get_vector_engine_async", getter)
    monkeypatch.setattr(
        job_module, "get_vectordb_context_config", lambda: {"vector_db_provider": "lancedb"}
    )
    try:
        for slot in range(4):
            await adapter.upsert_raw_vectors(
                "Maintenance_label",
                [{"id": uuid4(), "vector": [0.1, 0.2, 0.3], "payload": {"slot": slot}}],
                payload_schema=_Payload,
            )

        (result,) = await run_maintenance(pipeline_name="cognify_pipeline")

        assert result.job == "vector_compaction"
        assert result.status == "completed", result
        assert result.counts["tasks_executed"] == 1
        assert result.counts["fragments_removed"] == 4
        table = await adapter.get_collection("Maintenance_label")
        assert await table.count_rows() == 4
    finally:
        await adapter.close()


@pytest.fixture
def community_adapter(monkeypatch):
    """Register a community adapter class the way ``use_vector_adapter`` does."""
    from cognee.infrastructure.databases.vector import VectorDBInterface
    from cognee.infrastructure.databases.vector.supported_databases import supported_databases

    def register(name, implements_compact):
        attrs = {}
        if implements_compact:

            async def compact(self, collection_name=None):
                return {}

            attrs["compact"] = compact
        adapter_class = type(f"Fake{name.title()}Adapter", (VectorDBInterface,), attrs)
        monkeypatch.setitem(supported_databases, name, adapter_class)
        monkeypatch.setattr(
            job_module, "get_vectordb_context_config", lambda: {"vector_db_provider": name}
        )
        return adapter_class

    return register


def test_a_community_adapter_that_implements_compact_is_compacted(
    vector_settings, monkeypatch, community_adapter
):
    vector_settings(vector_db_compaction_enabled="true")
    community_adapter("qdrant", implements_compact=True)
    _no_engine(monkeypatch)
    assert VectorCompactionJob().gate(CTX) is None


def test_a_community_adapter_with_the_inherited_no_op_is_skipped(
    vector_settings, monkeypatch, community_adapter
):
    vector_settings(vector_db_compaction_enabled="true")
    community_adapter("milvus", implements_compact=False)
    _no_engine(monkeypatch)
    assert VectorCompactionJob().gate(CTX) == REASON_BACKEND_UNSUPPORTED


def test_vector_compaction_is_store_scoped():
    assert VectorCompactionJob.scope == "store"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["pgvector", "neptune_analytics", "turso"])
async def test_a_non_lancedb_store_is_skipped_without_creating_an_engine(
    vector_settings, monkeypatch, provider
):
    """Test 6 at unit level: a cognify on another backend is untouched."""
    from cognee.modules.maintenance import run_maintenance

    vector_settings(vector_db_compaction_enabled="true")
    monkeypatch.setattr(
        job_module, "get_vectordb_context_config", lambda: {"vector_db_provider": provider}
    )
    _no_engine(monkeypatch)

    (result,) = await run_maintenance(pipeline_name="cognify_pipeline")

    assert (result.job, result.status, result.reason) == (
        "vector_compaction",
        "skipped",
        REASON_BACKEND_UNSUPPORTED,
    )


def test_a_pass_where_every_collection_failed_is_errored():
    result = result_from_stats(
        "vector_compaction", {"A": {"error": "boom"}, "B": {"error": "disk full"}}
    )
    assert result.status == "errored"
    assert result.counts["collection_errors"] == 2
    assert result.has_failures


def test_partial_failures_keep_the_earned_status_and_are_flagged():
    result = result_from_stats(
        "vector_compaction",
        {
            "A": {
                "planned_tasks": 1,
                "executed_tasks": 1,
                "fragments_removed": 3,
                "fragments_added": 1,
                "prune_error": "prune failed",
            },
            "B": {"error": "boom"},
        },
    )
    assert result.status == "completed"
    assert (result.counts["collection_errors"], result.counts["prune_errors"]) == (1, 1)
    assert result.has_failures


def test_a_clean_idle_pass_has_no_failures():
    result = result_from_stats("vector_compaction", {"A": {"planned_tasks": 0}})
    assert result.status == "already_completed"
    assert not result.has_failures


@pytest.mark.asyncio
async def test_the_resolver_matches_what_the_factory_builds(tmp_path, monkeypatch):
    """The gate asks the factory's resolver; it must name the class the factory builds."""
    pytest.importorskip("lancedb")
    from cognee.infrastructure.databases.vector.supported_databases import supported_databases

    factory = importlib.import_module("cognee.infrastructure.databases.vector.create_vector_engine")
    monkeypatch.setattr(factory, "get_embedding_engine", _FakeEmbeddingEngine)
    # The uncached builder: the public factory wraps what it builds in a cache proxy.
    build = factory._create_vector_engine.__wrapped__

    def built_class(provider, url=""):
        return type(
            build(
                vector_db_provider=provider,
                vector_db_url=url,
                vector_db_name="resolver",
                vector_db_port="",
                vector_db_key="",
                vector_dataset_database_handler="lancedb",
                vector_db_username="",
                vector_db_password="",
                vector_db_host="",
                vector_db_subprocess_enabled=False,
            )
        )

    assert factory.resolve_vector_adapter_class("lancedb") is built_class(
        "lancedb", str(tmp_path / "db")
    )

    class RegisteredAdapter:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(supported_databases, "registered_store", RegisteredAdapter)
    assert factory.resolve_vector_adapter_class("registered_store") is built_class(
        "registered_store"
    )
    assert factory.resolve_vector_adapter_class("no_such_provider") is None
