"""A CSV through plain add()/cognify() must take the DLT route via the loader engine.

CSV handling moved out of resolve_dlt_sources into the loader system:
``dlt_csv_loader`` (registered above the text-flattening ``csv_loader``) runs
staging ingestion inside the ingest pipeline and returns the manifest as a
``LoaderResult`` carrying the stable ``data_id`` and the ``system_metadata``
route stamp. This drives the whole path end to end, hermetically: local CSV,
mocked embeddings (the DLT route makes no LLM calls).
"""

import json
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

DATASET = "csv_loader_ds"
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
        yield str(csv_file)

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_csv_takes_dlt_route_through_loader(clean_env):
    csv_path = clean_env

    await cognee.add([csv_path], dataset_name=DATASET)

    user = await get_default_user()
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]
    records = await get_dataset_data(dataset.id)
    manifests = [record for record in records if is_dlt_source_manifest(record)]
    assert len(manifests) == 1, f"CSV must become one manifest record, got {len(manifests)}"
    manifest = manifests[0]
    assert manifest.system_metadata.get("row_count") == 3
    assert manifest.loader_engine == "dlt_csv_loader"

    # raw_data_location holds the manifest text the DLT cognify route reads.
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(manifest.raw_data_location) as manifest_file:
        manifest_body = json.loads(manifest_file.read())
    assert len(manifest_body["rows"]) == 3

    # Re-adding the identical CSV must not mint a second record.
    await cognee.add([csv_path], dataset_name=DATASET)
    records_after = await get_dataset_data(dataset.id)
    manifests_after = [r for r in records_after if is_dlt_source_manifest(r)]
    assert len(manifests_after) == 1
    assert manifests_after[0].id == manifest.id

    await cognee.cognify(datasets=[DATASET])

    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    census: dict = {}
    for _, props in nodes:
        census[props.get("type", "?")] = census.get(props.get("type", "?"), 0) + 1

    assert census.get("DltRow") == 3, f"expected 3 DltRow nodes, census: {census}"
    assert census.get("SchemaTable") == 1, f"schema node missing: {census}"
