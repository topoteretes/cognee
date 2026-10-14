"""DLT deletions must never precede a confirmed foreground pipeline completion."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
    PipelineRunYield,
)

add_module = importlib.import_module("cognee.api.v1.add.add")


@pytest.fixture
def ingestion(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="documents")
    user = SimpleNamespace(id=uuid4())
    events = []

    async def cleanup():
        events.append("cleanup")

    orphan_cleanup = AsyncMock(side_effect=cleanup)
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: None
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    monkeypatch.setattr(
        add_module, "resolve_authorized_user_dataset", AsyncMock(return_value=(user, dataset))
    )
    monkeypatch.setattr(
        add_module,
        "resolve_dlt_sources",
        AsyncMock(return_value=(["replacement"], orphan_cleanup)),
    )
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())
    materialize = AsyncMock(return_value=["replacement"])
    monkeypatch.setattr(add_module, "materialize_stream_for_background", materialize)

    def run_info(kind):
        return kind(pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name)

    return SimpleNamespace(
        dataset=dataset,
        events=events,
        cleanup=orphan_cleanup,
        materialize=materialize,
        run_info=run_info,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [PipelineRunCompleted, PipelineRunAlreadyCompleted])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_foreground_cleanup_only_after_success(ingestion, monkeypatch, kind, wrapped):
    completed = ingestion.run_info(kind)

    async def execute(**kwargs):
        ingestion.cleanup.assert_not_awaited()
        ingestion.events.append("committed")
        return {ingestion.dataset.id: completed} if wrapped else completed

    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: execute)

    result = await add_module.add("source", dataset_name="documents")

    assert result is completed
    assert ingestion.events == ["committed", "cleanup"]
    ingestion.cleanup.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [PipelineRunErrored, PipelineRunStarted, PipelineRunYield, None])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_foreground_without_success_preserves_previous_data(
    ingestion, monkeypatch, kind, wrapped
):
    result = ingestion.run_info(kind) if kind else None
    response = {ingestion.dataset.id: result} if wrapped else result
    executor = AsyncMock(return_value=response)
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: executor)

    assert await add_module.add("source", dataset_name="documents") is result
    ingestion.cleanup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_pipeline_exception_or_cancellation_preserves_previous_data(
    ingestion, monkeypatch, background, error
):
    executor = AsyncMock(side_effect=error("ingestion failed"))
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: executor)

    with pytest.raises(error):
        await add_module.add("source", run_in_background=background)

    ingestion.cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_materialization_failure_preserves_previous_data(ingestion):
    ingestion.materialize.side_effect = OSError("stream unavailable")

    with pytest.raises(OSError, match="stream unavailable"):
        await add_module.add("source", run_in_background=True)

    ingestion.cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_pipeline_result_preserves_previous_data(ingestion, monkeypatch):
    executor = AsyncMock(return_value={})
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kwargs: executor)

    assert await add_module.add("source") == {}
    ingestion.cleanup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [PipelineRunCompleted, PipelineRunErrored])
async def test_detached_pipeline_never_cleans_up_even_after_return(ingestion, monkeypatch, kind):
    """Drive the real background executor through a late failure or completion."""
    finish = asyncio.Event()
    started = ingestion.run_info(PipelineRunStarted)
    final = ingestion.run_info(kind)

    async def pipeline(**kwargs):
        ingestion.cleanup.assert_not_awaited()
        yield started
        await finish.wait()
        ingestion.cleanup.assert_not_awaited()
        yield final

    monkeypatch.setattr(add_module, "run_pipeline", pipeline)
    queue = Mock()
    monkeypatch.setattr(pipeline_execution_mode, "push_to_queue", queue)
    previous_tasks = set(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS)

    try:
        result = await add_module.add("source", run_in_background=True)
        assert result is started
        ingestion.cleanup.assert_not_awaited()
    finally:
        tasks = pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS - previous_tasks
        finish.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    queue.assert_called_once_with(final.pipeline_run_id, final)
    ingestion.cleanup.assert_not_awaited()
