"""A row hard-deleted upstream must vanish from ALL per-dataset stores
(relational, graph, vector) after a re-sync, under multi-user access control.

Mechanism, under stable manifest identity: an explicit re-ingest
(add(..., incremental_loading=False, data_cache=False) — plain add() is
idempotent and skips completed items; update() is the UUID-based
alternative) updates the SAME manifest Data record in place, and the
deleted row's derived artifacts are purged by
``purge_stale_dlt_source_artifacts`` at the head of the DLT route during
re-cognify — inside the run's per-dataset DB context, so the purge hits
the per-dataset graph + vector stores, not the default engines.
(Historically this scenario went through ``_delete_dlt_orphans`` at add
time: a changed source produced a NEW content-addressed id and orphaned the
old manifest. Stable ids removed that churn; ``_delete_dlt_orphans`` now
only handles sources that disappear entirely.)

Local Ladybug + LanceDB, mocked LLM + ``MOCK_EMBEDDING`` — no live
credentials, no network.
"""

import hashlib
import pathlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import (
    graph_db_config,
    set_database_global_context_variables,
    vector_db_config,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

DATASET = "dlt_purge_ac_ds"


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    # The point of this regression: exercise the per-dataset (multi-user) path.
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.setenv("GRAPH_DATASET_DATABASE_HANDLER", "ladybug")
    monkeypatch.setenv("VECTOR_DATASET_DATABASE_HANDLER", "lancedb")
    monkeypatch.setenv("LLM_API_KEY", "sk-mocked")
    # Offline embeddings: MOCK_EMBEDDING makes the (default LiteLLM) engine return
    # canned zero-vectors — real vector rows land in LanceDB with no network call.
    monkeypatch.setenv("MOCK_EMBEDDING", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.set_graph_db_config(
        {"graph_database_provider": "ladybug", "graph_dataset_database_handler": "ladybug"}
    )
    cognee.config.set_vector_db_config(
        {"vector_db_provider": "lancedb", "vector_dataset_database_handler": "lancedb"}
    )
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.set_migration_db_config({"migration_db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()
    yield
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _mock_llm():
    """Patch the LLM's structured output with a canned graph + summary."""
    from unittest.mock import patch

    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    async def _out(text_input, system_prompt, response_model, **kw):
        name = getattr(response_model, "__name__", "")
        if name == "KnowledgeGraph":
            h = hashlib.md5((text_input or "").encode()).hexdigest()[:8]
            return KnowledgeGraph(
                nodes=[
                    Node(id=f"c_{h}_a", name=f"c_{h}_a", type="Concept", description="x"),
                    Node(id=f"c_{h}_b", name=f"c_{h}_b", type="Concept", description="x"),
                ],
                edges=[
                    Edge(
                        source_node_id=f"c_{h}_a",
                        target_node_id=f"c_{h}_b",
                        relationship_name="rel",
                    )
                ],
            )
        if name == "SummarizedContent":
            return SummarizedContent(summary=(text_input or "")[:120], description="")
        return response_model()

    return patch(
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output",
        side_effect=_out,
    )


def _dlt_source(rows):
    import dlt

    @dlt.resource(
        name="widgets",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def widgets():
        yield from rows

    return widgets


async def _get_dataset(user):
    return (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]


async def _dlt_pks(dataset):
    """Primary keys of all live DLT rows: manifest records (source ==
    "dlt_source") carry their rows in the manifest JSON; legacy per-row
    records (source == "dlt") are read directly."""
    from cognee.tasks.ingestion.dlt_utils import load_dlt_manifest

    rows = await get_dataset_data(dataset.id)
    pks = []
    for d in rows:
        ext = d.system_metadata if isinstance(d.system_metadata, dict) else {}
        if ext.get("source") == "dlt_source":
            manifest = await load_dlt_manifest(d.raw_data_location)
            pks.extend(row["primary_key_value"] for row in manifest.get("rows", []))
        elif ext.get("source") == "dlt":
            pks.append(ext.get("primary_key_value"))
    return sorted(pks)


async def _store_counts(dataset):
    """Count per-dataset graph nodes + DltRow vector rows (DLT rows live in
    their own DltRow_text collection; chunk search is documents-only). Under
    access control the graph/vector engines are per-dataset, so read them
    inside the dataset DB context."""
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        nodes, _ = await (await get_graph_engine()).get_graph_data()
        ve = await get_vector_engine_async()
        try:
            vec = await (await ve.get_collection("DltRow_text")).count_rows()
        except Exception:
            vec = 0
    return len(nodes), vec


@pytest.mark.asyncio
async def test_deleted_row_purged_from_per_dataset_stores_on_resync(clean_env):
    user = await get_default_user()
    kwargs = dict(primary_key="id", write_disposition="merge", max_rows_per_table=0)

    with _mock_llm():
        # Ingest two rows through the real add + cognify pipeline (per-dataset DB).
        await cognee.add(
            _dlt_source(
                [
                    {
                        "id": "a",
                        "body": "alpha runbook restart the payments service",
                        "_deleted": False,
                    },
                    {
                        "id": "b",
                        "body": "beta onboarding request vpn access from it",
                        "_deleted": False,
                    },
                ]
            ),
            dataset_name=DATASET,
            **kwargs,
        )
        await cognee.cognify(datasets=[DATASET])

        dataset = await _get_dataset(user)
        assert await _dlt_pks(dataset) == ["a", "b"]
        nodes_before, vec_before = await _store_counts(dataset)
        assert nodes_before > 0 and vec_before == 2  # graph populated, 2 chunks

        # Delete 'b' upstream and re-sync via an EXPLICIT re-ingest (plain
        # add() is idempotent — completed items keep the fast skip). The
        # manifest keeps its STABLE Data id: the record updates in place and
        # pipeline_status clears, so nothing is orphaned and the source is
        # never absent from the relational store.
        await cognee.add(
            _dlt_source([{"id": "b", "_deleted": True}]),
            dataset_name=DATASET,
            incremental_loading=False,
            data_cache=False,
            **kwargs,
        )
        # Re-cognify: purge_stale_dlt_source_artifacts drops the source's
        # previous graph/vector artifacts (including row 'b') inside the
        # run's per-dataset DB context, then the surviving row is re-emitted
        # (deterministic DLT pipeline — no LLM involved).
        await cognee.cognify(datasets=[DATASET])

    # 'b' must be gone from ALL three stores — including the per-dataset graph +
    # vector, which is exactly what the ledger-only / no-context path missed.
    dataset = await _get_dataset(user)
    assert await _dlt_pks(dataset) == ["a"]  # relational (shared store)
    nodes_after, vec_after = await _store_counts(dataset)
    assert vec_after == 1, (
        f"per-dataset vector not purged: {vec_before} -> {vec_after} (expected 1)"
    )
    assert nodes_after < nodes_before, (
        f"per-dataset graph not purged: {nodes_before} -> {nodes_after}"
    )


@pytest.mark.asyncio
async def test_legacy_content_addressed_manifest_purged_on_first_resync(clean_env):
    """The one manifest job _delete_dlt_orphans still owns: upgrade cleanup.

    Pre-stable-id deployments hold manifests whose Data id embeds a content
    hash. On the first re-add after upgrade, the fresh manifest gets the
    stable id, the legacy record's id is no longer in the fresh set, and the
    orphan path purges it — manifest, graph nodes, and vectors — inside the
    per-dataset context. This pins the "one final orphan-cycle reprocess, no
    manual step" migration promise.
    """
    import sys
    from unittest.mock import patch

    import dlt

    # Own dataset + source names: dlt keeps process-global pipeline state
    # keyed by name, so tests in one process must not share identities.
    legacy_dataset = "dlt_purge_legacy_ds"

    def _legacy_source(source_rows):
        @dlt.resource(
            name="widgets_legacy",
            primary_key="id",
            write_disposition="merge",
            columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
        )
        def widgets_legacy():
            yield from source_rows

        return widgets_legacy

    resolve_module = sys.modules["cognee.tasks.ingestion.resolve_dlt_sources"]
    real_get_unique_data_id = resolve_module.get_unique_data_id

    async def _legacy_style_id(identifier, user):
        # Emulate the retired scheme: identity varies with content.
        if identifier.startswith("dlt_source:"):
            identifier = f"{identifier}:deadbeefcafebabe"
        return await real_get_unique_data_id(identifier, user)

    user = await get_default_user()
    kwargs = dict(primary_key="id", write_disposition="merge", max_rows_per_table=0)
    rows = [
        {"id": "a", "body": "alpha runbook restart the payments service", "_deleted": False},
        {"id": "b", "body": "beta onboarding request vpn access from it", "_deleted": False},
    ]

    with _mock_llm():
        # 1. Ingest under the LEGACY id scheme and cognify — a pre-upgrade store.
        with patch.object(resolve_module, "get_unique_data_id", _legacy_style_id):
            await cognee.add(_legacy_source(rows), dataset_name=legacy_dataset, **kwargs)
        await cognee.cognify(datasets=[legacy_dataset])

        dataset = (
            await get_authorized_existing_datasets(
                user=user, permission_type="read", datasets=[legacy_dataset]
            )
        )[0]
        legacy_ids = {d.id for d in await get_dataset_data(dataset.id)}
        nodes_before, vec_before = await _store_counts(dataset)
        assert vec_before == 2, "pre-upgrade store must hold both row vectors"

        # 2. First re-add after 'upgrade' (unpatched, stable id): the legacy
        # manifest is orphaned and purged; the stable-id manifest replaces it.
        await cognee.add(_legacy_source(rows), dataset_name=legacy_dataset, **kwargs)
        await cognee.cognify(datasets=[legacy_dataset])

    records = await get_dataset_data(dataset.id)
    assert len(records) == 1, f"expected exactly the stable-id manifest, got {len(records)}"
    assert records[0].id not in legacy_ids, "the legacy-id manifest must be gone"
    assert await _dlt_pks(dataset) == ["a", "b"], "rows survive under the new identity"

    nodes_after, vec_after = await _store_counts(dataset)
    assert vec_after == 2, "row vectors rebuilt under the stable id, no leftovers"
    assert nodes_after <= nodes_before, "no orphaned graph artifacts accumulate"
