"""Background mode with per-item routing: a mixed dataset under
run_in_background=True returns an immediate handle and completes detached,
with the manifest taking the DLT route and the document the standard route —
one cognify_pipeline run, one terminal status, both artifact families present.

The reviewer's verification checklist named background execution explicitly;
the resolver rides the background executor as a plain parameter, and this
pins it end-to-end. Offline: LLM and embeddings are mocked.
"""

import asyncio
import pathlib
from unittest.mock import patch

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunStarted
from cognee.modules.users.methods import get_default_user

DATASET = "background_mixed_ds"
COMPLETION_TIMEOUT_SECONDS = 120


async def _mock_embed_text(self, texts):
    return [[float(len(t) % 7) / 10.0] * self.dimensions for t in texts]


async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
        return KnowledgeGraph(
            nodes=[
                Node(id="zorblatt", name="Zorblatt", type="Company", description="manufacturer")
            ],
            edges=[
                Edge(source_node_id="zorblatt", target_node_id="zorblatt", relationship_name="is")
            ],
        )
    if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
        return SummarizedContent(summary="Mock summary.", description="")
    if response_model is str:
        return "mock"
    return response_model()


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

    note = root / "note.txt"
    note.write_text("Zorblatt Industries manufactures underwater bicycles in Reykjavik.")

    with (
        patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text),
        patch.object(LLMGateway, "acreate_structured_output", new=_mock_acreate),
    ):
        yield str(note)

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _dlt_source():
    import dlt

    @dlt.resource(name="widgets", primary_key="id", write_disposition="replace")
    def widgets():
        yield from [{"id": "1", "name": "Widget"}, {"id": "2", "name": "Gadget"}]

    return widgets()


async def _cognify_run_statuses(dataset_id):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models.PipelineRun import PipelineRun

    async with get_relational_engine().get_async_session() as session:
        runs = (await session.execute(select(PipelineRun))).scalars().all()
    return [
        str(run.status).split(".")[-1]
        for run in runs
        if run.pipeline_name == "cognify_pipeline" and str(run.dataset_id) == str(dataset_id)
    ]


@pytest.mark.asyncio
async def test_background_cognify_routes_mixed_dataset(clean_env):
    note_path = clean_env
    await cognee.add([_dlt_source(), note_path], dataset_name=DATASET)

    user = await get_default_user()
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[DATASET]
        )
    )[0]

    # Background call returns immediately with a started handle per dataset.
    result = await cognee.cognify(datasets=[DATASET], run_in_background=True)
    handles = list(result.values())
    assert len(handles) == 1
    assert isinstance(handles[0], PipelineRunStarted)

    # The detached drain finishes on its own; poll the run record.
    deadline = asyncio.get_event_loop().time() + COMPLETION_TIMEOUT_SECONDS
    statuses = []
    while asyncio.get_event_loop().time() < deadline:
        statuses = await _cognify_run_statuses(dataset.id)
        if "DATASET_PROCESSING_COMPLETED" in statuses:
            break
        await asyncio.sleep(1)
    assert "DATASET_PROCESSING_COMPLETED" in statuses, (
        f"background run never completed; statuses={statuses}"
    )
    # Exactly one cognify run lifecycle for the dataset — never one per route.
    assert statuses.count("DATASET_PROCESSING_STARTED") <= 1

    # Both routes produced their artifacts inside that single background run.
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        graph = await get_graph_engine()
        nodes, _ = await graph.get_graph_data()
    census: dict = {}
    for _, props in nodes:
        census[props.get("type", "?")] = census.get(props.get("type", "?"), 0) + 1

    assert census.get("DltRow") == 2, f"DLT route artifacts missing: {census}"
    assert census.get("SchemaTable") == 1, f"schema node missing: {census}"
    assert census.get("Entity", 0) >= 1, f"standard-route artifacts missing: {census}"
