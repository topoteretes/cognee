"""An executor given to one ``cognify()`` or ``remember()`` call.

``data_item_executor`` may run that call's data items in other processes (see
``cognee.modules.pipelines.execution``). It is for the call it is given to: it
reaches ``run_tasks`` through the pipeline, and a remote instance cannot be
sent one.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

import cognee.modules.pipelines.operations.pipeline as pipeline_module
from cognee.infrastructure.locks import held_datasets
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
remember_module = importlib.import_module("cognee.api.v1.remember.remember")
serve_state_module = importlib.import_module("cognee.api.v1.serve.state")


async def _pipeline_call_of(**cognify_kwargs):
    """What ``cognify()`` calls the pipeline with."""
    calls = []

    async def fake_executor(**kwargs):
        calls.append(kwargs)
        return {"ds": "run_info"}

    with (
        patch.object(
            cognify_module, "get_pipeline_executor", lambda run_in_background: fake_executor
        ),
        patch.object(cognify_module, "get_default_tasks", new=AsyncMock(return_value="STANDARD")),
        patch.object(cognify_module, "get_dlt_tasks", new=AsyncMock(return_value="DLT_TASKS")),
        patch.object(cognify_module, "get_code_file_tasks", new=MagicMock(return_value="CODE")),
        patch.object(cognify_module, "get_code_repo_tasks", new=MagicMock(return_value="REPO")),
        patch.object(cognify_module, "get_indexing_tasks", new=AsyncMock(return_value="INDEX")),
        patch.object(
            cognify_module, "get_configured_ontology_resolver", new=MagicMock(return_value=None)
        ),
    ):
        await cognify_module.cognify(datasets=["ds"], **cognify_kwargs)
    (call,) = calls
    return call


@pytest.mark.asyncio
async def test_cognify_hands_its_executor_to_the_pipeline():
    executor = object()

    call = await _pipeline_call_of(data_item_executor=executor)

    assert call["data_item_executor"] is executor


@pytest.mark.asyncio
async def test_a_call_without_one_hands_the_pipeline_none():
    """So the executor registered for the process, if any, is what applies."""
    call = await _pipeline_call_of()

    assert call["data_item_executor"] is None


@pytest.mark.asyncio
async def test_the_pipeline_hands_the_executor_to_every_datasets_run(monkeypatch):
    datasets = [SimpleNamespace(id=uuid4(), name=name) for name in ("a", "b")]
    executor = object()
    handed = []

    async def run_pipeline_per_dataset(dataset, **kwargs):
        handed.append(kwargs["data_item_executor"])
        yield PipelineRunCompleted(
            pipeline_run_id=uuid4(), dataset_id=dataset.id, dataset_name=dataset.name
        )

    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), datasets)),
    )
    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", run_pipeline_per_dataset)

    async for _event in pipeline_module.run_pipeline(
        tasks=lambda item: [], data_item_executor=executor
    ):
        pass

    assert handed == [executor, executor]


@pytest.mark.asyncio
async def test_a_datasets_run_hands_the_executor_to_run_tasks(monkeypatch):
    dataset = SimpleNamespace(id=uuid4(), name="ds")
    executor = object()
    handed = []

    def run_tasks(*args, **kwargs):
        handed.append(kwargs["data_item_executor"])

        async def events():
            yield "event"

        return events()

    monkeypatch.setattr(pipeline_module, "run_tasks", run_tasks)
    monkeypatch.setattr(pipeline_module, "get_dataset_data", AsyncMock(return_value=[]))
    # As inside a run that already holds this dataset's lock.
    token = held_datasets.set(held_datasets.get() | {dataset.id})
    try:
        events = [
            event
            async for event in pipeline_module.run_pipeline_per_dataset(
                dataset=dataset,
                user=SimpleNamespace(id=uuid4()),
                tasks=lambda item: [],
                data_item_executor=executor,
            )
        ]
    finally:
        held_datasets.reset(token)

    assert events == ["event"]
    assert handed == [executor]


@pytest.mark.asyncio
async def test_cognify_rejects_the_argument_in_remote_mode(monkeypatch):
    """An executor is an object of this process: a remote instance cannot be sent it."""
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())

    with pytest.raises(ValueError, match="data_item_executor is not supported"):
        await cognify_module.cognify(data_item_executor=object())


def test_remember_routes_the_argument_to_cognify():
    assert "data_item_executor" in remember_module._COGNIFY_ONLY
    assert "data_item_executor" in remember_module.RememberKwargs.__annotations__


@pytest.mark.asyncio
async def test_remember_rejects_the_argument_for_a_session(monkeypatch):
    """The session path runs no pipeline over data items, so the executor would
    be dropped without a word."""
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: None)
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )

    with pytest.raises(ValueError, match="data_item_executor is not supported when session_id"):
        await remember_module.remember("some text", session_id="s1", data_item_executor=object())


@pytest.mark.asyncio
async def test_remember_rejects_the_argument_in_remote_mode(monkeypatch):
    monkeypatch.setattr(serve_state_module, "get_remote_client", lambda: MagicMock())

    with pytest.raises(ValueError, match="data_item_executor is not supported"):
        await remember_module.remember("some text", data_item_executor=object())
