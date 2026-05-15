from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


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


@pytest.mark.asyncio
@pytest.mark.parametrize("build_global_context_index", [False, True])
async def test_improve_global_context_index_opt_in(monkeypatch, build_global_context_index):
    import cognee.shared.utils as shared_utils

    improve_module = import_module("cognee.api.v1.improve.improve")
    serve_state = import_module("cognee.api.v1.serve.state")
    memify_module = import_module("cognee.modules.memify")
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(improve_module, "new_span", lambda _: DummySpan())

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

    assert result == {"status": "memify-ok"}
    memify_mock.assert_awaited_once()
    if build_global_context_index:
        global_context_mock.assert_awaited_once_with(
            user=user,
            dataset="docs",
            run_in_background=False,
        )
    else:
        global_context_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_improve_skips_global_context_index_in_background(monkeypatch):
    import cognee.shared.utils as shared_utils

    improve_module = import_module("cognee.api.v1.improve.improve")
    serve_state = import_module("cognee.api.v1.serve.state")
    memify_module = import_module("cognee.modules.memify")
    pipeline_module = import_module("cognee.memify_pipelines.global_context_index")
    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(improve_module, "new_span", lambda _: DummySpan())

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

    assert result == {"status": "memify-ok"}
    memify_mock.assert_awaited_once()
    global_context_mock.assert_not_awaited()


def test_improve_payload_global_context_index_defaults_to_false():
    router_module = import_module("cognee.api.v1.improve.routers.get_improve_router")

    payload = router_module.ImprovePayloadDTO(dataset_name="docs")
    enabled_payload = router_module.ImprovePayloadDTO(
        dataset_name="docs",
        build_global_context_index=True,
    )

    assert payload.build_global_context_index is False
    assert enabled_payload.build_global_context_index is True
