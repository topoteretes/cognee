"""Regression: DLT orphan cleanup must purge the *per-dataset* graph + vector
stores under multi-user access control, not just the relational store.

``_delete_dlt_orphans`` forgets a deleted DLT row's graph nodes + vector
embeddings via ``delete_data_nodes_and_edges``, which resolves the graph/vector
engines from the *ambient* async context. Under ``ENABLE_BACKEND_ACCESS_CONTROL``
those engines are per-dataset, and the cleanup call path does not always run
inside the dataset's DB context — the background-ingest path (``add()`` with
``run_in_background=True``) invokes ``orphan_cleanup`` *before* any pipeline
establishes the context. Without an explicit dataset-context wrapper the purge
silently targets the default engines and the forgotten row's chunks/entities
stay in the per-dataset graph + vector stores (still searchable).

This reproduces that fresh-context condition (reset the graph/vector context
vars, mirroring the background path) and asserts the row is purged from the
per-dataset graph + vector stores. Local Ladybug + LanceDB, mocked LLM +
``MOCK_EMBEDDING`` — no live credentials, no network.
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
        ext = d.external_metadata if isinstance(d.external_metadata, dict) else {}
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
async def test_dlt_orphan_cleanup_purges_graph_and_vector_under_access_control(clean_env):
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

        # Delete 'b' upstream and re-sync through the real add pipeline. Under
        # the manifest model the whole source is one content-addressed Data
        # record: the re-sync commits a fresh manifest (rows: only 'a') and the
        # foreground add then runs the deferred orphan cleanup, which purges
        # the previous manifest (rows: a, b) and all its derived artifacts.
        # Cleanup runs *after* the pipeline's dataset context has exited — the
        # exact condition the dataset-context wrapper in _delete_dlt_orphans
        # exists for; without it the graph + vector purge would resolve the
        # default engines and miss the per-dataset stores.
        await cognee.add(
            _dlt_source([{"id": "b", "_deleted": True}]), dataset_name=DATASET, **kwargs
        )
        # Re-cognify the fresh manifest so the surviving row's chunk is rebuilt
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
