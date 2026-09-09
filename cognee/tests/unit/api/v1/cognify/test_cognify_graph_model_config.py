import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from cognee.api.v1.config.config import config
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.data_models import KnowledgeGraph

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
cognify_package = importlib.import_module("cognee.api.v1.cognify")
cognify_router_module = importlib.import_module("cognee.api.v1.cognify.routers.get_cognify_router")
migrations_module = importlib.import_module("cognee.modules.migrations.startup")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


class ConfiguredGraphModel(BaseModel):
    name: str


class ExplicitGraphModel(BaseModel):
    title: str


@pytest.fixture
def restore_graph_model():
    graph_config = get_graph_config()
    original_graph_model = graph_config.graph_model
    try:
        yield
    finally:
        graph_config.graph_model = original_graph_model


async def _graph_model_passed_to_default_tasks(monkeypatch, **cognify_kwargs):
    get_default_tasks = AsyncMock(return_value=[])

    async def execute_pipeline(**kwargs):
        return {}

    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: None)
    monkeypatch.setattr(migrations_module, "run_migrations_and_block", AsyncMock())
    monkeypatch.setattr(cognify_module, "get_configured_ontology_resolver", lambda _: None)
    monkeypatch.setattr(cognify_module, "get_default_tasks", get_default_tasks)
    monkeypatch.setattr(cognify_module, "get_dlt_tasks", AsyncMock(return_value=[]))
    monkeypatch.setattr(cognify_module, "get_code_file_tasks", MagicMock(return_value=[]))
    monkeypatch.setattr(cognify_module, "get_code_repo_tasks", MagicMock(return_value=[]))
    monkeypatch.setattr(
        cognify_module,
        "get_pipeline_executor",
        lambda run_in_background: execute_pipeline,
    )

    await cognify_module.cognify(
        datasets=["dataset"],
        chunk_size=1,
        config={"ontology_config": {"ontology_resolver": None}},
        **cognify_kwargs,
    )

    return get_default_tasks.await_args.kwargs["graph_model"]


@pytest.mark.asyncio
async def test_cognify_uses_configured_graph_model_when_argument_is_omitted(
    monkeypatch, restore_graph_model
):
    config.set_graph_model(ConfiguredGraphModel)

    graph_model = await _graph_model_passed_to_default_tasks(monkeypatch)

    assert graph_model is ConfiguredGraphModel


@pytest.mark.asyncio
async def test_cognify_explicit_graph_model_overrides_configuration(
    monkeypatch, restore_graph_model
):
    config.set_graph_model(ConfiguredGraphModel)

    graph_model = await _graph_model_passed_to_default_tasks(
        monkeypatch, graph_model=ExplicitGraphModel
    )

    assert graph_model is ExplicitGraphModel


@pytest.mark.asyncio
async def test_cognify_uses_knowledge_graph_when_configuration_is_default(
    monkeypatch, restore_graph_model
):
    config.set_graph_model(KnowledgeGraph)

    graph_model = await _graph_model_passed_to_default_tasks(monkeypatch)

    assert graph_model is KnowledgeGraph


def test_cognify_router_preserves_omitted_graph_model(monkeypatch):
    captured = {}
    dataset_id = uuid4()

    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid4()), email="default@example.com", is_active=True, tenant_id=None
        )

    async def fake_cognify(*args, **kwargs):
        captured["graph_model"] = kwargs["graph_model"]
        return {
            dataset_id: PipelineRunCompleted(
                pipeline_run_id=uuid4(),
                dataset_id=dataset_id,
                dataset_name="dataset",
            )
        }

    monkeypatch.setattr(cognify_router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cognify_package, "cognify", fake_cognify)

    app = FastAPI()
    app.include_router(cognify_router_module.get_cognify_router(), prefix="/cognify")
    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user

    with TestClient(app) as client:
        response = client.post("/cognify", json={"datasets": ["dataset"]})

    assert response.status_code == 200
    assert captured["graph_model"] is None
