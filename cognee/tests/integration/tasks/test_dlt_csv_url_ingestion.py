"""A CSV given as an fsspec URL must auto-detect into the DLT route.

The old implementation ran every CSV path through os.path.abspath, which
mangles URLs ("s3://bucket/x.csv" -> "<cwd>/s3:/bucket/x.csv") and silently
pointed dlt's filesystem reader at a nonexistent local directory. This test
drives the URL branch end to end with file:// (hermetic — same code path as
s3://, only the fsspec backend differs; the real-S3 run lives in the CI S3
bucket test). Embeddings are mocked; the DLT route makes no LLM calls.
"""

import pathlib
from unittest.mock import patch

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
from cognee.tasks.ingestion.dlt_utils import is_dlt_source_manifest

DATASET = "csv_url_ds"
CSV_CONTENT = "id,name,section\n1,anemometer,weather\n2,barometer,weather\n3,seismograph,geology\n"


async def _mock_embed_text(self, texts):
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

    csv_file = root / "instruments.csv"
    csv_file.write_text(CSV_CONTENT)

    with patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text):
        yield f"file://{csv_file}"

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_csv_url_takes_dlt_route(clean_env):
    csv_url = clean_env

    await cognee.add([csv_url], dataset_name=DATASET)

    user = await get_default_user()
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]
    records = await get_dataset_data(dataset.id)
    manifests = [record for record in records if is_dlt_source_manifest(record)]
    assert len(manifests) == 1, f"URL CSV must become one manifest, got {len(manifests)}"
    assert manifests[0].system_metadata.get("row_count") == 3

    await cognee.cognify(datasets=[DATASET])

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    census: dict = {}
    for _, props in nodes:
        census[props.get("type", "?")] = census.get(props.get("type", "?"), 0) + 1

    assert census.get("DltRow") == 3, f"expected 3 DltRow nodes, census: {census}"
    assert census.get("SchemaTable") == 1, f"schema node missing: {census}"
