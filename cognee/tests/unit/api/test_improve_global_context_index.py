from contextlib import asynccontextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.improve import GraphCapabilities, ImproveResult
from cognee.modules.improve.stages import REASON_TURNED_OFF


class DummySpan:
    def __init__(self):
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.asyncio
async def test_global_context_index_pipeline_calls_memify(monkeypatch):
    from cognee.memify_pipelines import global_context_index as pipeline_module

    memify_mock = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(pipeline_module, "run_memify_tasks", memify_mock)

    user = SimpleNamespace(id="user-id")

    result = await pipeline_module.global_context_index_pipeline(
        user=user,
        dataset="docs",
        method="llm",
        max_bucket_size=7,
        placement_distance_threshold=0.8,
        rebuild=True,
        bucketing_strategy="graph",
        min_overlap=0.2,
    )

    assert result == {"status": "ok"}
    memify_mock.assert_awaited_once()
    call_kwargs = memify_mock.await_args.kwargs
    assert call_kwargs["dataset"] == "docs"
    assert call_kwargs["data"] == [{}]
    assert call_kwargs["user"] is user
    assert call_kwargs["run_in_background"] is False
    assert len(call_kwargs["extraction_tasks"]) == 1
    assert len(call_kwargs["enrichment_tasks"]) == 1
    task_kwargs = call_kwargs["enrichment_tasks"][0].default_params["kwargs"]
    assert task_kwargs["bucketing_strategy"] == "graph"
    assert task_kwargs["min_overlap"] == 0.2


def test_default_method_builds_the_deterministic_task_pair():
    from cognee.memify_pipelines.global_context_index import (
        get_global_context_index_memify_tasks,
    )

    extraction_tasks, enrichment_tasks = get_global_context_index_memify_tasks()

    assert [task.executable.__name__ for task in extraction_tasks] == ["load_dataset_content"]
    assert [task.executable.__name__ for task in enrichment_tasks] == ["build_document_index"]
    # No LLM task, so the run skips the first-run LLM connection probe.
    assert not any(task.needs_llm for task in extraction_tasks + enrichment_tasks)

    with pytest.raises(ValueError, match="from_entities"):
        get_global_context_index_memify_tasks(method="hierarchical")


def _stub_orchestrator(monkeypatch, resolved):
    """Stub everything improve() touches below the stage loop."""
    import cognee.shared.utils as shared_utils

    improve_module = import_module("cognee.api.v1.improve.improve")
    serve_state = import_module("cognee.api.v1.serve.state")
    startup = import_module("cognee.modules.migrations.startup")

    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(improve_module, "new_span", lambda _: DummySpan())
    monkeypatch.setattr(startup, "run_migrations_and_block", AsyncMock(return_value=None))
    monkeypatch.setattr(
        improve_module,
        "resolve_authorized_user_datasets",
        AsyncMock(side_effect=lambda dataset, user: (user, [resolved])),
    )
    monkeypatch.setattr(
        improve_module,
        "resolve_graph_capabilities",
        AsyncMock(return_value=GraphCapabilities.assume_supported()),
    )

    @asynccontextmanager
    async def fake_record_operation(name):
        from cognee.modules.operations.record_operation import OperationContext

        yield OperationContext(name)

    monkeypatch.setattr(improve_module, "record_operation", fake_record_operation)
    return improve_module


@pytest.mark.asyncio
@pytest.mark.parametrize("build_global_context_index", [False, True])
async def test_improve_global_context_index_opt_in(monkeypatch, build_global_context_index):
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    resolved = SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    improve_module = _stub_orchestrator(monkeypatch, resolved)

    global_context_mock = AsyncMock(return_value={"status": "global-context-ok"})
    monkeypatch.setattr(pipeline_module, "global_context_index_pipeline", global_context_mock)

    user = SimpleNamespace(id="user-id")

    result = await improve_module.improve(
        dataset="docs",
        user=user,
        build_global_context_index=build_global_context_index,
    )

    assert isinstance(result, ImproveResult)
    stage = result.stage("global_context_index")
    if build_global_context_index:
        # The stage receives the resolved dataset id, never the name.
        global_context_mock.assert_awaited_once_with(
            user=user,
            dataset=resolved.id,
            run_in_background=False,
        )
        assert stage.status == "completed"
    else:
        global_context_mock.assert_not_awaited()
        assert stage.status == "skipped"
        assert stage.reason == REASON_TURNED_OFF


@pytest.mark.asyncio
async def test_improve_background_runs_global_context_index_in_the_chain(monkeypatch):
    """Background mode no longer drops stage 9: the whole chain runs as one task."""
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    resolved = SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    improve_module = _stub_orchestrator(monkeypatch, resolved)

    global_context_mock = AsyncMock(return_value={"status": "global-context-ok"})
    monkeypatch.setattr(pipeline_module, "global_context_index_pipeline", global_context_mock)

    result = await improve_module.improve(
        dataset="docs",
        user=SimpleNamespace(id="user-id"),
        run_in_background=True,
        build_global_context_index=True,
        review_conflicts=False,
    )

    assert result.status == "running"
    await result.wait()

    assert result.status == "completed"
    global_context_mock.assert_awaited_once()
    assert global_context_mock.await_args.kwargs["run_in_background"] is False
    assert result.stage("global_context_index").status == "completed"


def test_improve_payload_global_context_index_defaults_to_the_server_config():
    router_module = import_module("cognee.api.v1.improve.routers.get_improve_router")

    payload = router_module.ImprovePayloadDTO(dataset_name="docs")
    enabled_payload = router_module.ImprovePayloadDTO(
        dataset_name="docs",
        build_global_context_index=True,
    )

    # None: the server's IMPROVE_GLOBAL_CONTEXT_INDEX decides; an explicit value wins.
    assert payload.build_global_context_index is None
    assert enabled_payload.build_global_context_index is True


@pytest.mark.asyncio
@pytest.mark.parametrize("config_default", [True, False])
async def test_improve_global_context_index_follows_the_config_when_not_given(
    monkeypatch, config_default
):
    """No argument means IMPROVE_GLOBAL_CONTEXT_INDEX decides (default true)."""
    from cognee.modules.improve.config import ImproveConfig

    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    resolved = SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    improve_module = _stub_orchestrator(monkeypatch, resolved)
    monkeypatch.setattr(
        improve_module,
        "get_improve_config",
        lambda: ImproveConfig(global_context_index=config_default),
    )
    global_context_mock = AsyncMock(return_value={"status": "global-context-ok"})
    monkeypatch.setattr(pipeline_module, "global_context_index_pipeline", global_context_mock)

    result = await improve_module.improve(dataset="docs", user=SimpleNamespace(id="user-id"))

    stage = result.stage("global_context_index")
    if config_default:
        global_context_mock.assert_awaited_once()
        assert stage.status == "completed"
    else:
        global_context_mock.assert_not_awaited()
        assert stage.reason == REASON_TURNED_OFF


def test_global_context_index_is_on_by_default():
    from cognee.modules.improve.config import ImproveConfig

    assert ImproveConfig(_env_file=None).global_context_index is True
