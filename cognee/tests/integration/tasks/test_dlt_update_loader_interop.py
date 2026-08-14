"""Row-level update() interop with the loader-era ingestion paths.

Two crossings the base features don't cover on their own:

1. A CSV ingested via the dlt_csv_loader (file -> loader -> manifest) is later
   updated through update(manifest_id, changed_csv_path). The update converts
   the path resolver-style; the identity seed (dataset, csv_source_name) must
   match the loader's, and only the delta may be reprocessed.
2. A mixed dataset (dlt resource + loader-ingested CSV): updating ONE source
   must leave the sibling manifest byte-identical — record, graph rows, and
   embeddings all untouched.

Embedding mock records every embedded text (ground truth for "what got
reprocessed"). Real add/update/cognify against local stores; the DLT route
makes no LLM calls.
"""

import pathlib
from unittest.mock import patch
from uuid import UUID

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

# dlt keeps process-global pipeline state keyed by name — every test in this
# process needs its own dataset name (and per-dataset source states).
DATASET_CSV = "upd_csv_loader_ds"
DATASET_MIX = "upd_mixed_ds"

EMBEDDED: list[str] = []

CSV_V1 = (
    "id,name,role\n"
    "1,Ada Lovelace,analytical engines\n"
    "2,Alan Turing,computability\n"
    "3,Grace Hopper,compilers\n"
)
# id=1 unchanged, id=2 edited, id=3 removed, id=4 added
CSV_V2 = (
    "id,name,role\n"
    "1,Ada Lovelace,analytical engines\n"
    "2,Alan Turing,computing pioneer\n"
    "4,John von Neumann,stored programs\n"
)


async def _mock_embed_text(self, texts):
    EMBEDDED.extend(texts)
    return [[float(len(t) % 7) / 10.0 + 0.1] * self.dimensions for t in texts]


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
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

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text):
        yield root

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _resource(rows, name):
    import dlt

    @dlt.resource(name=name, primary_key="id", write_disposition="replace")
    def source():
        yield from rows

    return source()


async def _manifests(user, dataset_name):
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[dataset_name]
        )
    )[0]
    manifests = {
        d.system_metadata.get("source_name"): d
        for d in await get_dataset_data(dataset.id)
        if isinstance(d.system_metadata, dict) and d.system_metadata.get("source") == "dlt_source"
    }
    return dataset, manifests


async def _dlt_row_nodes():
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    return {
        str(props.get("text", "")): str(node_id)
        for node_id, props in nodes
        if props.get("type") == "DltRow"
    }


@pytest.mark.asyncio
async def test_update_of_loader_ingested_csv_applies_delta(clean_env):
    """CSV in via the loader, updated via update(path): identities must line
    up across the two ingestion mechanisms and only the delta reprocesses."""
    root = clean_env
    user = await get_default_user()

    csv_path = root / "engineers.csv"
    csv_path.write_text(CSV_V1)

    await cognee.add([str(csv_path)], dataset_name=DATASET_CSV)
    await cognee.cognify(datasets=[DATASET_CSV])

    dataset, manifests = await _manifests(user, DATASET_CSV)
    manifest = manifests["engineers"]
    assert manifest.loader_engine == "dlt_csv_loader"

    rows_before = await _dlt_row_nodes()
    assert len(rows_before) == 3
    unchanged_id_before = next(nid for t, nid in rows_before.items() if "analytical" in t)

    csv_path.write_text(CSV_V2)
    EMBEDDED.clear()
    summary = await cognee.update(
        data_id=UUID(str(manifest.id)),
        data=str(csv_path),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )

    assert summary["rows_added"] == 1
    assert summary["rows_edited"] == 1
    assert summary["rows_removed"] == 1
    assert summary["rows_unchanged"] == 1

    rows_after = await _dlt_row_nodes()
    texts_after = " ".join(rows_after)
    assert "computing pioneer" in texts_after, "edited value must be in the graph"
    assert "stored programs" in texts_after, "added row must be in the graph"
    assert "compilers" not in texts_after, "removed row must be gone"
    assert "computability" not in texts_after, "pre-edit value must be gone"
    unchanged_id_after = next(nid for t, nid in rows_after.items() if "analytical" in t)
    assert unchanged_id_after == unchanged_id_before, "unchanged row keeps its node id"

    # Only the delta got embedded: the unchanged row's text must not reappear.
    assert not any("analytical" in t for t in EMBEDDED), "unchanged row was re-embedded"
    assert any("computing pioneer" in t for t in EMBEDDED)
    assert any("stored programs" in t for t in EMBEDDED)

    # The synced manifest must satisfy a later cognify without reprocessing.
    EMBEDDED.clear()
    await cognee.cognify(datasets=[DATASET_CSV])
    assert not any("DltRow" in t or "Ada" in t for t in EMBEDDED), (
        "cognify after update must skip the already-synced manifest"
    )


@pytest.mark.asyncio
async def test_update_leaves_sibling_sources_untouched(clean_env):
    """Updating one source in a mixed dataset (dlt resource + loader CSV)
    must not touch the sibling manifest's record, rows, or embeddings."""
    root = clean_env
    user = await get_default_user()

    csv_path = root / "instruments.csv"
    csv_path.write_text("id,kind\n1,anemometer\n2,barometer\n")

    await cognee.add(
        [
            _resource(
                [{"id": "1", "status": "active"}, {"id": "2", "status": "pending"}], "orders"
            ),
            str(csv_path),
        ],
        dataset_name=DATASET_MIX,
        primary_key="id",
    )
    await cognee.cognify(datasets=[DATASET_MIX])

    dataset, manifests = await _manifests(user, DATASET_MIX)
    assert set(manifests) == {"orders", "instruments"}
    sibling_before = manifests["instruments"]
    sibling_hash = str(sibling_before.content_hash)
    sibling_status = dict(sibling_before.pipeline_status)

    rows_before = await _dlt_row_nodes()
    sibling_row_ids = {
        nid for t, nid in rows_before.items() if "anemometer" in t or "barometer" in t
    }
    assert len(sibling_row_ids) == 2

    EMBEDDED.clear()
    summary = await cognee.update(
        data_id=UUID(str(manifests["orders"].id)),
        data=_resource(
            [{"id": "1", "status": "active"}, {"id": "2", "status": "shipped"}], "orders"
        ),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )
    assert summary["rows_edited"] == 1

    _, manifests_after = await _manifests(user, DATASET_MIX)
    sibling_after = manifests_after["instruments"]
    assert str(sibling_after.content_hash) == sibling_hash, "sibling record must not change"
    assert dict(sibling_after.pipeline_status) == sibling_status, (
        "sibling pipeline status must survive the update"
    )

    rows_after = await _dlt_row_nodes()
    sibling_ids_after = {
        nid for t, nid in rows_after.items() if "anemometer" in t or "barometer" in t
    }
    assert sibling_ids_after == sibling_row_ids, "sibling graph rows must be untouched"
    assert not any("anemometer" in t or "barometer" in t for t in EMBEDDED), (
        "sibling rows must not be re-embedded by the update"
    )
