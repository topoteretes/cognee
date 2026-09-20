"""A started item must report a terminal event when its consumer cancels or closes it."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

telemetry_module = importlib.import_module(
    "cognee.modules.pipelines.operations.run_tasks_with_telemetry"
)


@pytest.fixture
def telemetry(monkeypatch):
    emit = Mock()
    monkeypatch.setattr(telemetry_module, "send_telemetry", emit)
    monkeypatch.setattr(telemetry_module, "get_current_settings", dict)
    return emit


def _run():
    return telemetry_module.run_tasks_with_telemetry(
        tasks=[], data=[], user=SimpleNamespace(tenant_id=None), pipeline_name="test_pipeline"
    )


def _events(telemetry):
    return [call.args[0] for call in telemetry.call_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [[], ["first", "second"]])
async def test_completed_pipeline_emits_one_terminal_event(monkeypatch, telemetry, results):
    async def run_base(*args):
        for result in results:
            yield result

    monkeypatch.setattr(telemetry_module, "run_tasks_base", run_base)
    assert [result async for result in _run()] == results
    assert _events(telemetry) == ["Pipeline Run Started", "Pipeline Run Completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("yield_first", [False, True])
async def test_failed_pipeline_preserves_exception(monkeypatch, telemetry, yield_first):
    error = ValueError("task failed")

    async def run_base(*args):
        if yield_first:
            yield "first"
        raise error

    monkeypatch.setattr(telemetry_module, "run_tasks_base", run_base)
    with pytest.raises(ValueError) as exc_info:
        async for _ in _run():
            pass
    assert exc_info.value is error
    assert _events(telemetry) == ["Pipeline Run Started", "Pipeline Run Errored"]


@pytest.mark.asyncio
async def test_cancelled_consumer_emits_error_and_still_cancels(monkeypatch, telemetry):
    started = asyncio.Event()
    finished = asyncio.Event()

    async def run_base(*args):
        try:
            started.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            finished.set()

    async def consume():
        async for _ in _run():
            pass

    monkeypatch.setattr(telemetry_module, "run_tasks_base", run_base)
    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=2)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert consumer.cancelled()
    assert finished.is_set()
    assert _events(telemetry) == ["Pipeline Run Started", "Pipeline Run Errored"]


@pytest.mark.asyncio
async def test_closing_partial_pipeline_emits_error_once(monkeypatch, telemetry):
    finished = asyncio.Event()

    async def run_base(*args):
        try:
            yield "first"
            yield "second"
        finally:
            finished.set()

    monkeypatch.setattr(telemetry_module, "run_tasks_base", run_base)
    pipeline = _run()
    assert await anext(pipeline) == "first"
    await pipeline.aclose()
    await pipeline.aclose()
    assert finished.is_set()
    assert _events(telemetry) == ["Pipeline Run Started", "Pipeline Run Errored"]


@pytest.mark.asyncio
async def test_closing_unstarted_pipeline_emits_nothing(telemetry):
    await _run().aclose()
    assert _events(telemetry) == []
