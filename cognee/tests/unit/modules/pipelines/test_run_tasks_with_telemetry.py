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


@pytest.fixture
def pipeline_logger(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(telemetry_module, "logger", logger)
    return logger


def _run():
    return telemetry_module.run_tasks_with_telemetry(
        tasks=[], data=[], user=SimpleNamespace(tenant_id=None), pipeline_name="test_pipeline"
    )


STARTED = telemetry_module.PIPELINE_ITEM_STARTED
COMPLETED = telemetry_module.PIPELINE_ITEM_COMPLETED
ERRORED = telemetry_module.PIPELINE_ITEM_ERRORED


def _events(telemetry):
    return [call.args[0] for call in telemetry.call_args_list]


def _errored_exception_type(telemetry):
    (errored,) = [call for call in telemetry.call_args_list if call.args[0] == ERRORED]
    return errored.kwargs["additional_properties"]["exception_type"]


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [[], ["first", "second"]])
async def test_completed_pipeline_emits_one_terminal_event(monkeypatch, telemetry, results):
    async def run_base(*args):
        for result in results:
            yield result

    monkeypatch.setattr(telemetry_module, "run_tasks_base", run_base)
    assert [result async for result in _run()] == results
    assert _events(telemetry) == [STARTED, COMPLETED]


@pytest.mark.asyncio
@pytest.mark.parametrize("yield_first", [False, True])
async def test_failed_pipeline_preserves_exception(
    monkeypatch, telemetry, pipeline_logger, yield_first
):
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
    assert _events(telemetry) == [STARTED, ERRORED]
    pipeline_logger.exception.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_consumer_emits_error_and_still_cancels(
    monkeypatch, telemetry, pipeline_logger
):
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
    assert _events(telemetry) == [STARTED, ERRORED]
    pipeline_logger.exception.assert_not_called()
    pipeline_logger.info.assert_any_call(
        "Pipeline item cancelled: `%s` (%s)", "test_pipeline", "CancelledError"
    )
    assert _errored_exception_type(telemetry) == "CancelledError"


@pytest.mark.asyncio
async def test_closing_partial_pipeline_emits_error_once(monkeypatch, telemetry, pipeline_logger):
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
    # aclosing closes the inner generator right away, not when it is collected.
    assert finished.is_set()
    assert _events(telemetry) == [STARTED, ERRORED]
    pipeline_logger.exception.assert_not_called()
    pipeline_logger.info.assert_any_call(
        "Pipeline item cancelled: `%s` (%s)", "test_pipeline", "GeneratorExit"
    )
    assert _errored_exception_type(telemetry) == "GeneratorExit"


@pytest.mark.asyncio
async def test_closing_unstarted_pipeline_emits_nothing(telemetry):
    await _run().aclose()
    assert _events(telemetry) == []
