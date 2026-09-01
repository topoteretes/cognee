"""Eval-capture wirings (SDK-529): record_operation and run_tasks open a run
scope only when capture is active, the manifest carries the (late-bound)
dataset, and only the pipeline runner drains.

``record_operation``'s row writer is stubbed; ``run_tasks`` is driven through
``runner_plumbing``. No LLM, no network, no database.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.run_tasks as run_tasks_module
from cognee.modules.observability import capture
from cognee.modules.observability.capture import KIND_RUN_MANIFEST, KIND_SUMMARY_GENERATED

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")

pytestmark = pytest.mark.usefixtures("capture_reset")


@pytest.fixture
def no_row_writes(monkeypatch):
    monkeypatch.setattr(record_operation_mod, "_write_operation_row", AsyncMock())


@pytest.fixture
def drain_spy(monkeypatch):
    """Replace capture.drain with a spy; returns (spy, real_drain)."""
    real_drain = capture.drain
    spy = AsyncMock()
    monkeypatch.setattr(capture, "drain", spy)
    return spy, real_drain


def _manifests(sink):
    return [record for record in sink.records if record["kind"] == KIND_RUN_MANIFEST]


# ---------------------------------------------------------------------------
# 14: record_operation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_operation_opens_no_scope_when_capture_is_off(monkeypatch, no_row_writes):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)

    async with record_operation_mod.record_operation("search") as context:
        assert capture.is_active() is False
        assert capture.current_scope() is None
        context.set_dataset(uuid4())

    from cognee.modules.observability.capture import hook

    assert not hook._buffer
    assert not hook._flushers


@pytest.mark.asyncio
async def test_record_operation_emits_manifest_without_draining(
    no_row_writes, fake_capture_sink, drain_spy
):
    spy, real_drain = drain_spy
    dataset_id = uuid4()

    async with record_operation_mod.record_operation("search") as context:
        scope = capture.current_scope()
        assert scope is not None
        assert scope.kind == "operation"
        assert scope.run_id == context.operation_id
        assert scope.dataset_id is None
        capture.emit(KIND_SUMMARY_GENERATED, "inside", payload_kind="text")
        # Bound mid-body, the way recall/search do it.
        context.set_dataset(dataset_id)

    spy.assert_not_awaited()
    assert capture.current_scope() is None

    await real_drain()

    [manifest] = _manifests(fake_capture_sink)
    assert manifest["run_id"] == str(context.operation_id)
    assert manifest["dataset_id"] == str(dataset_id)
    assert manifest["payload"]["kind"] == "operation"
    assert manifest["payload"]["dataset_id"] == str(dataset_id)
    # The event buffered before set_dataset() picked the dataset up too.
    [event] = [r for r in fake_capture_sink.records if r["kind"] == KIND_SUMMARY_GENERATED]
    assert event["run_id"] == str(context.operation_id)
    assert event["dataset_id"] == str(dataset_id)


@pytest.mark.asyncio
async def test_record_operation_failure_still_emits_manifest_and_reraises(
    no_row_writes, fake_capture_sink, drain_spy
):
    spy, real_drain = drain_spy

    with pytest.raises(ValueError, match="boom"):
        async with record_operation_mod.record_operation("recall") as context:
            raise ValueError("boom")

    spy.assert_not_awaited()
    await real_drain()

    [manifest] = _manifests(fake_capture_sink)
    assert manifest["run_id"] == str(context.operation_id)
    assert manifest["dataset_id"] is None


# ---------------------------------------------------------------------------
# 15: run_tasks
# ---------------------------------------------------------------------------


async def _drive(run_tasks_module_, dataset):
    events = []
    async for event in run_tasks_module_.run_tasks(
        tasks=[],
        dataset_id=dataset.id,
        data=["a"],
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        pipeline_name="cognify_pipeline",
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_run_tasks_opens_pipeline_scope_and_drains_once_after_terminal_log(
    monkeypatch, runner_plumbing, fake_capture_sink
):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    logs = runner_plumbing(run_tasks_module, dataset)

    order = []
    logs.complete.side_effect = lambda *args, **kwargs: order.append("complete")
    real_drain = capture.drain

    async def spy_drain(timeout=5.0):
        order.append("drain")
        await real_drain(timeout)

    monkeypatch.setattr(capture, "drain", spy_drain)

    seen_scopes = []

    async def fake_item(*args, **kwargs):
        seen_scopes.append(capture.current_scope())
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", fake_item)

    events = await _drive(run_tasks_module, dataset)
    pipeline_run_id = events[0].pipeline_run_id

    assert order == ["complete", "drain"]
    [scope] = seen_scopes
    assert scope.kind == "pipeline"
    assert scope.run_id == pipeline_run_id
    assert scope.dataset_id == dataset.id
    assert scope.sampled is True
    assert scope.finished is True
    assert capture.current_scope() is None

    # The run's own drain covered its manifest (finish() ran before it); the
    # scope's exit — after the terminal yield — did not enqueue a duplicate.
    assert not capture.hook._buffer
    [manifest] = _manifests(fake_capture_sink)
    assert manifest["run_id"] == str(pipeline_run_id)
    assert manifest["dataset_id"] == str(dataset.id)
    assert manifest["payload"]["kind"] == "pipeline"
    assert manifest["payload"]["sampled"] is True
    await real_drain()
    assert len(_manifests(fake_capture_sink)) == 1


@pytest.mark.asyncio
async def test_run_tasks_drains_once_on_the_error_path(
    monkeypatch, runner_plumbing, fake_capture_sink
):
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    logs = runner_plumbing(run_tasks_module, dataset)

    order = []
    logs.error.side_effect = lambda *args, **kwargs: order.append("error")
    real_drain = capture.drain

    async def spy_drain(timeout=5.0):
        order.append("drain")
        await real_drain(timeout)

    monkeypatch.setattr(capture, "drain", spy_drain)

    async def failing_item(*args, **kwargs):
        raise RuntimeError("item exploded")

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", failing_item)

    with pytest.raises(RuntimeError, match="item exploded"):
        await _drive(run_tasks_module, dataset)

    assert order == ["error", "drain"]
    logs.complete.assert_not_awaited()
    # The error path finishes the scope before draining too.
    [manifest] = _manifests(fake_capture_sink)
    assert manifest["payload"]["kind"] == "pipeline"
    await real_drain()
    assert len(_manifests(fake_capture_sink)) == 1


@pytest.mark.asyncio
async def test_run_tasks_skips_capture_when_off(monkeypatch, runner_plumbing):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)
    dataset = SimpleNamespace(id=uuid4(), name="ds", owner_id=uuid4())
    runner_plumbing(run_tasks_module, dataset)

    spy = AsyncMock()
    monkeypatch.setattr(capture, "drain", spy)

    seen_scopes = []

    async def fake_item(*args, **kwargs):
        seen_scopes.append(capture.current_scope())
        return {"run_info": "ok"}

    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", fake_item)

    await _drive(run_tasks_module, dataset)

    assert seen_scopes == [None]
    spy.assert_not_awaited()
