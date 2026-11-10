"""memify, TRIPLET_COMPLETION and memify_run stay one release as deprecated shims (SDK-1115)."""

import importlib
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.search.types import SearchType
from cognee.modules.search.types.SearchType import resolve_deprecated_search_type
from cognee.modules.users.methods import get_authenticated_user

shim_module = importlib.import_module("cognee.api.v1.memify.memify")
runner_module = importlib.import_module("cognee.modules.memify.run_memify_tasks")
router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")


async def _passthrough(data):
    return data


# --- internal runner ---------------------------------------------------------


@pytest.fixture
def runner_executor(monkeypatch):
    """Stub the runner's setup, migrations and dataset resolution; capture the executor call."""
    dataset = SimpleNamespace(id=uuid.uuid4(), owner_id=uuid.uuid4())
    executor = AsyncMock(return_value={"ran": True})
    monkeypatch.setattr(runner_module, "setup", AsyncMock())
    monkeypatch.setattr("cognee.modules.migrations.startup.run_migrations_and_block", AsyncMock())
    monkeypatch.setattr(
        runner_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=("user", [dataset])),
    )
    monkeypatch.setattr(runner_module, "get_pipeline_executor", lambda **_: executor)
    return executor


@pytest.mark.asyncio
async def test_runner_refuses_an_empty_task_list(runner_executor):
    with pytest.raises(ValueError, match="at least one"):
        await runner_module.run_memify_tasks([], [])
    runner_executor.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "expected"),
    [(None, [{}]), ([], []), ([{"seed": 1}], [{"seed": 1}])],
)
async def test_runner_seeds_only_when_data_is_none(runner_executor, data, expected):
    await runner_module.run_memify_tasks(
        [Task(_passthrough)],
        [],
        data=data,
        vector_db_config={"v": 1},
        graph_db_config={"g": 1},
    )
    kwargs = runner_executor.await_args.kwargs
    assert kwargs["data"] == expected
    assert kwargs["vector_db_config"] == {"v": 1}
    assert kwargs["graph_db_config"] == {"g": 1}
    assert kwargs["pipeline_name"] == "memify_pipeline"


# --- public cognee.memify shim ---------------------------------------------


def test_cognee_memify_is_the_deprecated_shim():
    assert cognee.memify is shim_module.memify


@pytest.fixture
def legacy_env(monkeypatch):
    """Stub what the shim calls before the runner; record the order of the calls."""
    dataset = SimpleNamespace(id=uuid.uuid4(), owner_id=uuid.uuid4())
    order = []

    def recorder(name, value=None):
        async def call(*args, **kwargs):
            order.append(name)
            return value

        return AsyncMock(side_effect=call)

    @asynccontextmanager
    async def database_context(*_args):
        yield

    fragment = object()
    env = SimpleNamespace(
        dataset=dataset,
        order=order,
        fragment=fragment,
        resolve=recorder("resolve", ("user", [dataset])),
        get_fragment=recorder("fragment", fragment),
        runner=AsyncMock(return_value={"ran": True}),
    )
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", recorder("setup"))
    monkeypatch.setattr(
        "cognee.modules.migrations.startup.run_migrations_and_block", recorder("migrations")
    )
    monkeypatch.setattr(
        "cognee.modules.pipelines.layers.resolve_authorized_user_datasets."
        "resolve_authorized_user_datasets",
        env.resolve,
    )
    monkeypatch.setattr(
        "cognee.context_global_variables.set_database_global_context_variables",
        database_context,
    )
    monkeypatch.setattr(
        "cognee.modules.retrieval.utils.brute_force_triplet_search.get_memory_fragment",
        env.get_fragment,
    )
    monkeypatch.setattr("cognee.modules.memify.run_memify_tasks", env.runner)
    return env


@pytest.fixture
def run_memify(legacy_env):
    return legacy_env.runner


@pytest.mark.asyncio
async def test_memify_warns_and_runs_the_given_tasks(legacy_env, run_memify):
    task = Task(_passthrough)

    with pytest.warns(FutureWarning, match=r"cognee\.improve\(\)"):
        result = await cognee.memify(extraction_tasks=[task], data=["x"], dataset="ds")

    assert result == {"ran": True}
    args, kwargs = run_memify.await_args
    assert args == ([task], [])
    assert kwargs["data"] == ["x"]
    assert kwargs["dataset"] == legacy_env.dataset.id
    assert legacy_env.resolve.await_args.args == ("ds", None)


@pytest.mark.asyncio
async def test_memify_gates_on_migrations_before_touching_the_dataset(legacy_env):
    with pytest.warns(FutureWarning):
        await cognee.memify(extraction_tasks=[Task(_passthrough)], dataset="ds")
    assert legacy_env.order == ["setup", "migrations", "resolve", "fragment"]


@pytest.mark.asyncio
async def test_memify_without_tasks_runs_nothing_but_still_checks_the_dataset(
    legacy_env, run_memify
):
    with pytest.warns(FutureWarning):
        assert await cognee.memify(dataset="ds") == {}
    assert legacy_env.order == ["setup", "migrations", "resolve"]
    run_memify.assert_not_awaited()


@pytest.mark.asyncio
async def test_memify_without_tasks_still_rejects_an_unauthorized_dataset(legacy_env, run_memify):
    from cognee.modules.users.exceptions import PermissionDeniedError

    legacy_env.resolve.side_effect = PermissionDeniedError("no access")
    with pytest.warns(FutureWarning), pytest.raises(PermissionDeniedError):
        await cognee.memify(dataset=uuid.uuid4())
    run_memify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "message"),
    [("get_triplet_datapoints", "removed"), ("no_such_task", "Unknown memify task name")],
)
async def test_memify_rejects_removed_and_unknown_task_names(legacy_env, run_memify, name, message):
    with pytest.warns(FutureWarning), pytest.raises(CogneeValidationError, match=message):
        await cognee.memify(extraction_tasks=[name], data=["x"])
    run_memify.assert_not_awaited()
    assert legacy_env.order == []  # rejected before any database work


@pytest.mark.asyncio
async def test_memify_resolves_registry_names(run_memify):
    with pytest.warns(FutureWarning):
        await cognee.memify(extraction_tasks=["extract_subgraph_chunks"], data=["x"])
    [extraction], [] = run_memify.await_args.args
    assert extraction.executable.__name__ == "extract_subgraph_chunks"


@pytest.mark.asyncio
async def test_memify_without_data_passes_the_graph(legacy_env, run_memify):
    with pytest.warns(FutureWarning):
        await cognee.memify(
            extraction_tasks=[Task(_passthrough)], node_name=["Alice"], node_type=None
        )

    assert legacy_env.get_fragment.await_args.kwargs == {"node_type": None, "node_name": ["Alice"]}
    kwargs = run_memify.await_args.kwargs
    assert kwargs["data"] == [legacy_env.fragment]
    assert kwargs["dataset"] == legacy_env.dataset.id
    assert kwargs["user"] == "user"


@pytest.mark.asyncio
async def test_memify_seeds_a_graph_reading_extraction_instead_of_projecting(
    legacy_env, run_memify
):
    """extract_code_docs reads the code graph itself: no memory fragment is built for it."""
    with pytest.warns(FutureWarning):
        await cognee.memify(
            extraction_tasks=["extract_code_docs"],
            enrichment_tasks=["add_data_points", "add_code_doc_edges"],
        )

    legacy_env.get_fragment.assert_not_awaited()
    assert legacy_env.order == ["setup", "migrations", "resolve"]
    kwargs = run_memify.await_args.kwargs
    assert kwargs["data"] == [{}]
    assert kwargs["dataset"] == legacy_env.dataset.id


@pytest.mark.asyncio
async def test_old_module_path_is_the_same_deprecated_memify(legacy_env, run_memify):
    """Before SDK-1115 cognee.memify *was* cognee.modules.memify.memify."""
    from cognee.modules.memify import memify as old_path_memify

    task = Task(_passthrough)
    with pytest.warns(FutureWarning, match=r"cognee\.memify\(\) is deprecated") as record:
        await old_path_memify(extraction_tasks=[task], data=["x"])
    assert record[0].filename == __file__
    assert run_memify.await_args.args == ([task], [])


def test_old_task_registry_path_still_resolves_names():
    import sys

    sys.modules.pop("cognee.memify_pipelines.memify_task_registry", None)
    with pytest.warns(FutureWarning, match="memify_task_registry is deprecated"):
        registry = importlib.import_module("cognee.memify_pipelines.memify_task_registry")

    assert registry.resolve_memify_tasks(None) is None
    assert registry.resolve_memify_tasks([]) is None
    [task] = registry.resolve_memify_tasks(["extract_subgraph"])
    assert task.executable.__name__ == "extract_subgraph"
    assert "extract_subgraph" in registry.supported_memify_task_names()
    assert "get_triplet_datapoints" not in registry.supported_memify_task_names()


def test_memify_task_package_still_exports_the_subgraph_tasks():
    from cognee.tasks.memify import extract_subgraph, extract_subgraph_chunks

    assert callable(extract_subgraph) and callable(extract_subgraph_chunks)


# --- POST /api/v1/memify ---------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(router_module.get_memify_router(), prefix="/api/v1/memify")

    async def authenticated_user():
        return SimpleNamespace(id=str(uuid.uuid4()), email="u@example.com", tenant_id=None)

    monkeypatch.setattr(router_module, "send_telemetry", lambda *a, **k: None)
    app.dependency_overrides[get_authenticated_user] = authenticated_user
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def memify_stub(monkeypatch):
    stub = AsyncMock(return_value={"ran": True})
    monkeypatch.setattr("cognee.api.v1.memify.memify", stub)
    return stub


def test_route_is_marked_deprecated(client):
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/memify"]["post"]
    assert operation["deprecated"] is True


def test_route_needs_a_dataset(client, memify_stub):
    assert client.post("/api/v1/memify", json={}).status_code == 400
    memify_stub.assert_not_awaited()


def test_route_runs_the_shim(client, memify_stub):
    response = client.post(
        "/api/v1/memify",
        json={"datasetName": "ds", "extractionTasks": ["extract_subgraph"], "nodeName": ["A"]},
    )
    assert response.status_code == 200
    assert response.json() == {"ran": True}
    kwargs = memify_stub.await_args.kwargs
    assert kwargs["extraction_tasks"] == ["extract_subgraph"]
    assert kwargs["dataset"] == "ds"
    assert kwargs["node_name"] == ["A"]


def test_route_surfaces_the_task_error(client, memify_stub):
    memify_stub.return_value = PipelineRunErrored(
        pipeline_run_id=uuid.uuid4(), dataset_id=uuid.uuid4(), dataset_name="ds", payload="boom"
    )
    response = client.post("/api/v1/memify", json={"datasetName": "ds"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_route_rejects_a_removed_task_name(client, run_memify):
    response = client.post(
        "/api/v1/memify",
        json={"datasetName": "ds", "extractionTasks": ["get_triplet_datapoints"]},
    )
    assert response.status_code == 422
    run_memify.assert_not_awaited()


# --- cognee-cli memify -----------------------------------------------------


def test_cli_memify_warns_and_runs_nothing(monkeypatch):
    from cognee.cli._cognee import _discover_commands
    from cognee.cli.commands import memify_command

    assert memify_command.MemifyCommand in _discover_commands()
    shim = AsyncMock(return_value={})
    monkeypatch.setattr(cognee, "memify", shim)
    echo = MagicMock()
    monkeypatch.setattr(memify_command, "fmt", echo)

    args = SimpleNamespace(
        dataset_name="ds", dataset_id=None, data=None, node_name=["A"], background=False
    )
    memify_command.MemifyCommand().execute(args)

    echo.warning.assert_called_once()
    assert "cognee-cli improve" in echo.warning.call_args.args[0]
    echo.success.assert_not_called()
    shim.assert_not_awaited()


def test_cli_memify_over_the_api_warns_and_calls_no_server(monkeypatch):
    from cognee.cli import api_dispatch

    client = MagicMock()
    echo = MagicMock()
    monkeypatch.setattr(api_dispatch, "fmt", echo)

    api_dispatch._dispatch_memify(client, SimpleNamespace(dataset_name="ds", dataset_id=None))

    echo.warning.assert_called_once()
    echo.success.assert_not_called()
    client.memify.assert_not_called()


# --- SearchType.TRIPLET_COMPLETION -----------------------------------------


def test_triplet_completion_resolves_to_graph_completion():
    with pytest.warns(FutureWarning, match="TRIPLET_COMPLETION is deprecated"):
        assert (
            resolve_deprecated_search_type(SearchType.TRIPLET_COMPLETION)
            is SearchType.GRAPH_COMPLETION
        )


@pytest.mark.parametrize("query_type", [None, SearchType.GRAPH_COMPLETION, SearchType.CHUNKS])
def test_current_search_types_pass_through_silently(recwarn, query_type):
    assert resolve_deprecated_search_type(query_type) is query_type
    assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]


@pytest.fixture
def remote_client(monkeypatch):
    """A connected remote instance: the first dispatch after the type is resolved."""
    client = SimpleNamespace(search=AsyncMock(return_value=[]), recall=AsyncMock(return_value=[]))
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_search_runs_triplet_completion_as_graph_completion(remote_client):
    with pytest.warns(FutureWarning, match="TRIPLET_COMPLETION") as record:
        await cognee.search("q", query_type=SearchType.TRIPLET_COMPLETION)
    assert record[0].filename == __file__
    assert remote_client.search.await_args.kwargs["search_type"] is SearchType.GRAPH_COMPLETION


@pytest.mark.asyncio
async def test_recall_runs_triplet_completion_as_graph_completion(remote_client):
    with pytest.warns(FutureWarning, match="TRIPLET_COMPLETION") as record:
        await cognee.recall("q", query_type=SearchType.TRIPLET_COMPLETION)
    # Points at the caller, not at recall()'s telemetry decorator.
    assert record[0].filename == __file__
    assert remote_client.recall.await_args.args[1] is SearchType.GRAPH_COMPLETION
