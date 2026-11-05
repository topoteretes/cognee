import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared import utils
from cognee.tests.unit.modules.pipelines.test_run_tasks_rollback import _FakeEngine, _no_op_context

base = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_base")
runner = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")
items = importlib.import_module("cognee.modules.pipelines.operations.run_tasks_with_telemetry")


@pytest.mark.asyncio
async def test_error_reporting_preserves_original_exception(monkeypatch):
    def broken_emitter(*args, **kwargs):
        raise RuntimeError("telemetry builder failed")

    monkeypatch.setattr(utils, "send_telemetry", broken_emitter)

    @utils.telemetry_on_error("op ERRORED")
    async def op():
        raise PermissionError("original operation error")

    with pytest.raises(PermissionError):
        await op()


class ImmutableError(Exception):
    """Accept Python's exception bookkeeping, but forbid diagnostic attributes."""

    def __setattr__(self, name, value):
        if name not in {"__traceback__", "__cause__", "__context__", "__suppress_context__"}:
            raise AttributeError("exception attributes are immutable")
        super().__setattr__(name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, ImmutableError])
async def test_nested_error_is_reported_once_per_run_without_mutating_it(monkeypatch, error_type):
    events = []
    monkeypatch.setattr(
        base,
        "send_telemetry",
        lambda name, *a, **k: events.append((name, k["additional_properties"])),
    )
    failed_future = asyncio.get_running_loop().create_future()
    failure = error_type("upstream failure")
    failed_future.set_exception(failure)

    async def upstream(data):
        return data

    async def task(data):
        return await failed_future

    run_ids = [uuid4(), uuid4()]
    for run_id in run_ids:
        with pytest.raises(error_type) as raised:
            async for _ in base.run_tasks_base(
                [Task(upstream), Task(task)],
                [1],
                SimpleNamespace(id=uuid4(), tenant_id=None),
                PipelineContext(pipeline_run_id=run_id),
            ):
                pass
        assert raised.value is failure
    errors = [props for name, props in events if name == "Coroutine Task Errored"]
    assert [(p["pipeline_run_id"], p["task_name"], p["exception_type"]) for p in errors] == [
        (str(run_id), "task", error_type.__name__) for run_id in run_ids
    ]
    assert vars(failure) == {}


def test_item_profile_is_total_for_nonfinite_sizes():
    assert items.data_item_telemetry_properties([SimpleNamespace(data_size=float("inf"))]) == {
        "item_count": 1
    }


def wire_runner(monkeypatch, data=None):
    dataset = SimpleNamespace(id=uuid4(), name="dataset", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    monkeypatch.setattr(runner, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(runner, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(
        runner,
        "log_pipeline_run_start",
        AsyncMock(return_value=SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(runner, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(runner, "log_pipeline_run_progress", AsyncMock())
    monkeypatch.setattr(runner, "pipeline_run_telemetry_properties", lambda *a, **k: {})
    return runner.run_tasks(
        [Task(lambda x: x)], dataset.id, data if data is not None else [1], user
    )


@pytest.mark.asyncio
async def test_error_telemetry_cannot_skip_persisting_run_failure(monkeypatch):
    gen = wire_runner(monkeypatch)
    monkeypatch.setattr(
        runner, "run_tasks_data_item", AsyncMock(side_effect=ValueError("original"))
    )

    def emit(name, *a, **k):
        if name == "Pipeline Run Errored":
            raise RuntimeError("telemetry builder failed")

    monkeypatch.setattr(runner, "send_telemetry", emit)
    try:
        async for _ in gen:
            pass
    except (ValueError, RuntimeError):
        pass
    assert runner.log_pipeline_run_error.await_count == 1


@pytest.mark.asyncio
async def test_started_record_has_a_started_event_before_background_handoff(monkeypatch):
    gen = wire_runner(monkeypatch)
    events = []
    monkeypatch.setattr(runner, "send_telemetry", lambda name, *a, **k: events.append(name))
    await anext(gen)
    await gen.aclose()
    assert events == ["Pipeline Run Started", "Pipeline Run Errored"]
    assert runner.log_pipeline_run_error.await_count == 1


@pytest.mark.asyncio
async def test_reported_framework_matches_actual_gateway_dispatch(monkeypatch):
    settings = importlib.import_module("cognee.modules.settings.get_current_settings")
    gateway = importlib.import_module("cognee.infrastructure.llm.LLMGateway")
    native_module = importlib.import_module(
        "cognee.infrastructure.llm.structured_output_framework.litellm_native.get_native_client"
    )
    configured = SimpleNamespace(
        llm_provider="openai",
        llm_model="test",
        llm_api_key="test",
        structured_output_framework="instructor",
    )
    monkeypatch.setattr(settings, "get_llm_context_config", lambda: configured)
    monkeypatch.setattr(settings, "resolve_embedding_names", lambda *a: ("openai", "test"))
    global_config = SimpleNamespace(structured_output_framework="litellm_native")
    monkeypatch.setattr(gateway, "get_llm_config", lambda: global_config)
    monkeypatch.setattr(settings, "get_llm_config", lambda: global_config)
    monkeypatch.setattr(gateway, "_inject_agent_memory", lambda text: text)
    monkeypatch.setattr(gateway, "_record_session_usage_after", lambda inner, **k: inner)
    monkeypatch.setattr(gateway, "_fail_fast_on_quota", lambda inner: inner)
    native = AsyncMock(return_value={})
    monkeypatch.setattr(
        native_module,
        "get_native_client",
        lambda: SimpleNamespace(acreate_structured_output=native),
    )
    label = settings.get_current_settings()["llm"]["structured_output"]
    await gateway.LLMGateway.acreate_structured_output("text", "system", dict)
    assert native.await_count == 1
    assert label == "litellm_native"


@pytest.mark.asyncio
async def test_existing_recovery_respects_a_dataset_that_is_still_locked(monkeypatch):
    from cognee.infrastructure.locks import get_dataset_lock
    from cognee.modules.cognify import recovery
    from cognee.tests.unit.modules.cognify.test_recovery import _dataset_for, _run, _wire

    active_run = _run(
        "cognify_pipeline", run_info={"progress": {"completed_items": 9, "total_items": 10}}
    )
    dataset = _dataset_for(active_run)
    calls = _wire(monkeypatch, [active_run], {dataset.id: dataset})
    async with await get_dataset_lock(dataset.id):
        await recovery.recover_stale_pipeline_runs_on_startup()
    assert calls == []


@pytest.mark.asyncio
async def test_setup_failure_records_terminal_event(monkeypatch):
    from contextlib import asynccontextmanager

    gen = wire_runner(monkeypatch)
    events = []
    monkeypatch.setattr(runner, "send_telemetry", lambda name, *a, **k: events.append(name))

    @asynccontextmanager
    async def broken_context(*a, **k):
        raise ConnectionError("database setup failed")
        yield

    monkeypatch.setattr(runner, "set_database_global_context_variables", broken_context)
    with pytest.raises(ConnectionError):
        async for _ in gen:
            pass
    assert events == ["Pipeline Run Started", "Pipeline Run Errored"]
    assert runner.log_pipeline_run_error.await_count == 1


@pytest.mark.asyncio
async def test_error_property_failure_preserves_original_exception(monkeypatch):
    def broken_properties(error):
        raise ValueError("malformed diagnostic")

    monkeypatch.setattr(utils, "telemetry_exception_properties", broken_properties)

    @utils.telemetry_on_error("op ERRORED")
    async def operation():
        raise PermissionError("original")

    with pytest.raises(PermissionError, match="original"):
        await operation()


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 429.5, True])
def test_invalid_status_codes_are_omitted(value):
    error = ValueError("original")
    error.status_code = value
    assert utils.telemetry_exception_properties(error) == {"exception_type": "ValueError"}


def test_status_descriptor_failure_is_omitted():
    class Failure(Exception):
        @property
        def status_code(self):
            raise RuntimeError("unavailable")

    assert utils.telemetry_exception_properties(Failure()) == {"exception_type": "Failure"}


@pytest.mark.asyncio
async def test_sibling_items_stop_before_rollback(monkeypatch):
    gen = wire_runner(monkeypatch, data=[1, 2])
    running = asyncio.Event()
    stopped = asyncio.Event()
    rollback = []

    async def item(value, *args):
        if value == 1:
            await running.wait()
            raise ValueError("first item failed")
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def record_error(*args, **kwargs):
        rollback.append(stopped.is_set())

    monkeypatch.setattr(runner, "run_tasks_data_item", item)
    monkeypatch.setattr(runner, "log_pipeline_run_error", record_error)
    with pytest.raises(ValueError):
        async for _ in gen:
            pass
    assert rollback == [True]
