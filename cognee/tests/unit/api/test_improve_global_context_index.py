from contextlib import asynccontextmanager
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.improve import GraphCapabilities, ImproveResult
from cognee.modules.improve.stages import REASON_OPT_IN_DISABLED


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
    monkeypatch.setattr(pipeline_module, "memify", memify_mock)

    user = SimpleNamespace(id="user-id")

    result = await pipeline_module.global_context_index_pipeline(
        user=user,
        dataset="docs",
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


def _stub_orchestrator(monkeypatch, resolved):
    """Stub everything improve() touches below the stage loop."""
    import cognee.shared.utils as shared_utils

    improve_module = import_module("cognee.api.v1.improve.improve")
    serve_state = import_module("cognee.api.v1.serve.state")
    startup = import_module("cognee.modules.migrations.startup")
    cognify_config = import_module("cognee.modules.cognify.config")
    changes = import_module("cognee.modules.improve.graph_changes")

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
    # Enrichment on: the stage would otherwise skip with triplet_embedding_disabled.
    monkeypatch.setattr(
        cognify_config, "get_cognify_config", lambda: SimpleNamespace(triplet_embedding=True)
    )
    monkeypatch.setattr(
        changes, "has_graph_changed_since_last_improve", AsyncMock(return_value=True)
    )
    return improve_module


@pytest.mark.asyncio
@pytest.mark.parametrize("build_global_context_index", [False, True])
async def test_improve_global_context_index_opt_in(monkeypatch, build_global_context_index):
    memify_module = import_module("cognee.modules.memify")
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    resolved = SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    improve_module = _stub_orchestrator(monkeypatch, resolved)

    memify_mock = AsyncMock(return_value={"status": "memify-ok"})
    global_context_mock = AsyncMock(return_value={"status": "global-context-ok"})
    monkeypatch.setattr(memify_module, "memify", memify_mock)
    monkeypatch.setattr(pipeline_module, "global_context_index_pipeline", global_context_mock)

    user = SimpleNamespace(id="user-id")

    result = await improve_module.improve(
        dataset="docs",
        user=user,
        build_global_context_index=build_global_context_index,
    )

    assert isinstance(result, ImproveResult)
    assert result.memify_run == {"status": "memify-ok"}
    assert result.to_legacy_dict() == {"status": "memify-ok"}
    memify_mock.assert_awaited_once()
    stage = result.stage("global_context_index")
    if build_global_context_index:
        # The stage receives the resolved dataset id, never the name.
        global_context_mock.assert_awaited_once_with(
            user=user,
            dataset=resolved.id,
            run_in_background=False,
            bucketing_strategy="graph",
            max_bucket_size=4,
        )
        assert stage.status == "completed"
    else:
        global_context_mock.assert_not_awaited()
        assert stage.status == "skipped"
        assert stage.reason == REASON_OPT_IN_DISABLED


@pytest.mark.asyncio
async def test_improve_background_runs_global_context_index_in_the_chain(monkeypatch):
    """Background mode no longer drops stage 9: the whole chain runs as one task."""
    memify_module = import_module("cognee.modules.memify")
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    resolved = SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    improve_module = _stub_orchestrator(monkeypatch, resolved)

    memify_mock = AsyncMock(return_value={"status": "memify-ok"})
    global_context_mock = AsyncMock(return_value={"status": "global-context-ok"})
    monkeypatch.setattr(memify_module, "memify", memify_mock)
    monkeypatch.setattr(pipeline_module, "global_context_index_pipeline", global_context_mock)

    result = await improve_module.improve(
        dataset="docs",
        user=SimpleNamespace(id="user-id"),
        run_in_background=True,
        build_global_context_index=True,
    )

    assert result.status == "running"
    await result.wait()

    assert result.status == "completed"
    assert result.memify_run == {"status": "memify-ok"}
    memify_mock.assert_awaited_once()
    global_context_mock.assert_awaited_once()
    assert memify_mock.await_args.kwargs["run_in_background"] is False
    assert global_context_mock.await_args.kwargs["run_in_background"] is False
    assert result.stage("global_context_index").status == "completed"


def test_improve_payload_global_context_index_defaults_to_false():
    router_module = import_module("cognee.api.v1.improve.routers.get_improve_router")

    payload = router_module.ImprovePayloadDTO(dataset_name="docs")
    enabled_payload = router_module.ImprovePayloadDTO(
        dataset_name="docs",
        build_global_context_index=True,
    )

    assert payload.build_global_context_index is False
    assert enabled_payload.build_global_context_index is True
