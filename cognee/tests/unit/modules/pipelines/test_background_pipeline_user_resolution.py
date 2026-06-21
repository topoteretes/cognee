"""Guards dataset-owner resolution in the background pipeline runner.

When ``run_pipeline_as_background_process`` is called without explicit
``datasets``, it resolves "all datasets the user can write to". The user is
supposed to come from ``params`` (the normal case), falling back to the default
user only when none was passed. A regression bound ``user`` *only* inside the
``if not params.get("user")`` branch, so when a caller did pass a user the name
was never assigned and the function raised ``NameError`` before it could look up
the datasets — breaking background ``cognify``/``add`` over all datasets.
"""

import asyncio
from types import SimpleNamespace

import pytest

from cognee.modules.pipelines.layers import pipeline_execution_mode
from cognee.modules.pipelines.layers.pipeline_execution_mode import (
    run_pipeline_as_background_process,
)


class _FakeRunInfo:
    def __init__(self, dataset_id, payload=None, pipeline_run_id="run-1"):
        self.dataset_id = dataset_id
        self.payload = payload
        self.pipeline_run_id = pipeline_run_id


async def _drain_background_tasks():
    """Let any spawned background tasks finish so the loop stays clean."""
    for task in list(pipeline_execution_mode._BACKGROUND_PIPELINE_TASKS):
        await task
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_background_process_uses_user_from_params_when_no_datasets(monkeypatch):
    """A user passed in params must be used and must not raise NameError."""
    monkeypatch.setattr(pipeline_execution_mode, "push_to_queue", lambda *a, **k: None)

    sentinel_user = SimpleNamespace(id="user-1")
    authorized_dataset = SimpleNamespace(id="ds1")

    seen = {}

    async def fake_get_authorized_existing_datasets(datasets, permission, user):
        seen["user"] = user
        return [authorized_dataset]

    async def fake_get_default_user():
        seen["default_user_called"] = True
        return SimpleNamespace(id="default-user")

    monkeypatch.setattr(
        pipeline_execution_mode,
        "get_authorized_existing_datasets",
        fake_get_authorized_existing_datasets,
    )
    monkeypatch.setattr(pipeline_execution_mode, "get_default_user", fake_get_default_user)

    async def pipeline(**kwargs):
        yield _FakeRunInfo(dataset_id="ds1")

    # Before the fix this raises NameError: name 'user' is not defined.
    started = await run_pipeline_as_background_process(pipeline, datasets=None, user=sentinel_user)

    assert "ds1" in started
    # The user from params is what gets used...
    assert seen["user"] is sentinel_user
    # ...and the default-user fallback must not be triggered.
    assert "default_user_called" not in seen

    await _drain_background_tasks()


@pytest.mark.asyncio
async def test_background_process_falls_back_to_default_user(monkeypatch):
    """With no user in params, the default user is resolved and used."""
    monkeypatch.setattr(pipeline_execution_mode, "push_to_queue", lambda *a, **k: None)

    default_user = SimpleNamespace(id="default-user")
    authorized_dataset = SimpleNamespace(id="ds1")

    seen = {}

    async def fake_get_authorized_existing_datasets(datasets, permission, user):
        seen["user"] = user
        return [authorized_dataset]

    async def fake_get_default_user():
        return default_user

    monkeypatch.setattr(
        pipeline_execution_mode,
        "get_authorized_existing_datasets",
        fake_get_authorized_existing_datasets,
    )
    monkeypatch.setattr(pipeline_execution_mode, "get_default_user", fake_get_default_user)

    async def pipeline(**kwargs):
        yield _FakeRunInfo(dataset_id="ds1")

    started = await run_pipeline_as_background_process(pipeline, datasets=None)

    assert "ds1" in started
    assert seen["user"] is default_user

    await _drain_background_tasks()
