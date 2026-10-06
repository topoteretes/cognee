"""How a multi-dataset run tells the hook which dataset is last.

With multi-user off every dataset shares one store, so a store-scoped job runs
once, after the last dataset. ``run_pipeline`` (foreground) and the background
executor (one ``run_pipeline`` call per dataset) must agree on which dataset
that is, and the API must refuse to boot on a bad ``MAINTENANCE_*`` value.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.pipelines.layers.pipeline_execution_mode as execution_module
import cognee.modules.pipelines.operations.pipeline as pipeline_module
from cognee.modules.pipelines.layers.hook_for_dataset import hook_for_dataset


async def hook(**kwargs):
    return kwargs


def _last_flag(wrapped):
    """What ``last_in_invocation`` the hook will be called with (None = its default)."""
    if isinstance(wrapped, partial):
        return wrapped.keywords.get("last_in_invocation")
    return None


def test_hook_for_dataset_marks_all_but_the_last():
    assert [_last_flag(hook_for_dataset(hook, i, 3)) for i in range(3)] == [False, False, None]
    assert hook_for_dataset(hook, 0, 1) is hook
    assert hook_for_dataset(None, 0, 3) is None


def test_marks_compose_across_the_executor_and_run_pipeline():
    """The executor marks dataset 0 of 2; the single-dataset run_pipeline inside
    that call must not turn it back into 'last'."""
    outer = hook_for_dataset(hook, 0, 2)
    inner = hook_for_dataset(outer, 0, 1)
    assert _last_flag(inner) is False


@pytest.mark.asyncio
async def test_run_pipeline_marks_every_dataset_but_the_last(monkeypatch):
    datasets = [SimpleNamespace(id=uuid4(), name=f"ds{i}") for i in range(3)]
    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(
        pipeline_module,
        "resolve_authorized_user_datasets",
        AsyncMock(return_value=(SimpleNamespace(id=uuid4()), datasets)),
    )
    received = []

    async def fake_per_dataset(*, dataset, after_run_completed, **kwargs):
        received.append((dataset.name, _last_flag(after_run_completed)))
        yield SimpleNamespace(dataset_id=dataset.id)

    monkeypatch.setattr(pipeline_module, "run_pipeline_per_dataset", fake_per_dataset)

    async for _ in pipeline_module.run_pipeline(
        tasks=lambda item: [], datasets=[d.id for d in datasets], after_run_completed=hook
    ):
        pass

    assert received == [("ds0", False), ("ds1", False), ("ds2", None)]


@pytest.mark.asyncio
async def test_the_background_executor_marks_every_dataset_but_the_last():
    calls = []

    async def fake_pipeline(**params):
        calls.append((params["datasets"], _last_flag(params.get("after_run_completed"))))
        yield SimpleNamespace(dataset_id=params["datasets"], payload=None, pipeline_run_id=uuid4())

    await execution_module.run_pipeline_as_background_process(
        fake_pipeline, datasets=["a", "b", "c"], after_run_completed=hook
    )
    for task in list(execution_module._BACKGROUND_PIPELINE_TASKS):
        await task

    assert calls == [("a", False), ("b", False), ("c", None)]


@pytest.mark.asyncio
async def test_the_background_executor_adds_no_hook_to_pipelines_without_one():
    """It runs other pipeline functions too; they take no such parameter."""
    seen = []

    async def fake_pipeline(**params):
        seen.append(set(params))
        yield SimpleNamespace(dataset_id=params["datasets"], payload=None, pipeline_run_id=uuid4())

    await execution_module.run_pipeline_as_background_process(fake_pipeline, datasets=["a", "b"])
    for task in list(execution_module._BACKGROUND_PIPELINE_TASKS):
        await task

    assert all("after_run_completed" not in params for params in seen)


@pytest.mark.asyncio
async def test_the_api_refuses_to_start_with_a_bad_maintenance_setting(monkeypatch):
    """A MAINTENANCE_JOBS_DISABLED typo fails the boot, as IMPROVE_* values do."""
    import importlib

    from cognee.modules.maintenance import get_maintenance_config

    client_module = importlib.import_module("cognee.api.client")

    async def _noop(*args, **kwargs):
        return None

    # Startup work unrelated to settings (resolved as submodules: the packages
    # re-export these names).
    for module_name, attr in (
        ("cognee.run_migrations", "run_migrations"),
        ("cognee.modules.users.methods", "get_default_user"),
    ):
        monkeypatch.setattr(importlib.import_module(module_name), attr, _noop)
    monkeypatch.setenv("MAINTENANCE_JOBS_DISABLED", "vector_compation")
    get_maintenance_config.cache_clear()
    try:
        with pytest.raises(ValueError, match="vector_compation"):
            async with client_module.lifespan(client_module.app):
                pass
    finally:
        monkeypatch.delenv("MAINTENANCE_JOBS_DISABLED")
        get_maintenance_config.cache_clear()
