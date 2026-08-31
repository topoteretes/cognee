"""Functional tests for "situation 2": a per-item task failure must not be silent.

Drives the REAL run_tasks pipeline (relational DB, dataset row, run records)
with a task that raises per item. Before COG-6276 this path swallowed the
cause: the yielded run info and the pipeline_runs row said only "Pipeline run
failed. Data item could not be processed." — the silent first-build failure
that 854/854 churned accounts hit. Asserts the ROOT cause now survives to:

1. the yielded ``PipelineRunErrored`` (error_class / scrubbed error_message),
2. the persisted ``pipeline_runs`` row,
3. the typed ``CognifyFailedError`` cognify raises from that run info.

Lives in the integration tree because it exercises the real add() pipeline and
relational engine; the environment fixture scopes every write to a tmp root.
"""

import pathlib

import pytest
import pytest_asyncio
from sqlalchemy import select

import cognee
from cognee.infrastructure.databases.relational import create_db_and_tables, get_relational_engine
from cognee.modules.pipelines.exceptions import CognifyFailedError
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.modules.pipelines.operations.run_tasks import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user

# Named like the real provider exception so assertions read like production.
AuthenticationError = type("AuthenticationError", (Exception,), {})


async def failing_task(data):
    raise AuthenticationError("Incorrect API key provided: sk-test-situation-two")


@pytest_asyncio.fixture
async def clean_test_environment(request, tmp_path, monkeypatch):
    """Scope every write to a per-test tmp root; never touch ambient data."""
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")

    root = pathlib.Path(tmp_path) / request.node.name

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))

    await create_db_and_tables()

    yield

    create_relational_engine.cache_clear()


async def _stage_dataset(dataset_name):
    """add() creates the dataset + data rows without any LLM call."""
    user = await get_default_user()
    await cognee.add("situation two probe text", dataset_name=dataset_name, user=user)

    from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data

    [dataset] = await get_authorized_existing_datasets([dataset_name], "read", user)
    data_items = await get_dataset_data(dataset.id)
    assert data_items, "add() should have created a data row"
    return user, dataset, data_items


@pytest.mark.asyncio
async def test_silent_item_path_carries_root_cause(clean_test_environment, monkeypatch):
    # The silent path: per-item errors are collected instead of re-raised.
    # (With the default "true", the error re-raises out of the item worker
    # and takes the run-level path instead — that variant is covered below.)
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "false")

    user, dataset, data_items = await _stage_dataset("situation_two")

    run_infos = []
    async for run_info in run_tasks(
        [Task(failing_task)],
        dataset_id=dataset.id,
        data=data_items,
        user=user,
        pipeline_name="cognify_pipeline",
        # The per-item (incremental) worker is the path with the silent-catch
        # behavior this test pins; the regular path always re-raises.
        incremental_loading=True,
    ):
        run_infos.append(run_info)

    # Situation 2 completes the generator (no raise) and yields an errored
    # run info — which must now carry the ROOT cause, not the generic wrapper.
    errored = [info for info in run_infos if isinstance(info, PipelineRunErrored)]
    assert errored, f"expected an errored run info, got {run_infos}"
    final = errored[-1]
    assert final.error_class == "AuthenticationError", final.error_class
    assert "Incorrect API key" in (final.error_message or ""), final.error_message
    assert "Pipeline run failed" not in (final.error_message or "")

    # The persisted run record carries the same root cause (was NULL before).
    async with get_relational_engine().get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).where(
                        PipelineRun.dataset_id == dataset.id,
                        PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_ERRORED,
                    )
                )
            )
            .scalars()
            .all()
        )
    assert rows, "expected an ERRORED pipeline_runs row"
    assert rows[-1].error_class == "AuthenticationError", rows[-1].error_class
    assert "Incorrect API key" in (rows[-1].error_message or "")

    # And cognify's raiser converts that run info into the typed loud error.
    from cognee.api.v1.cognify.cognify import raise_if_cognify_errored

    with pytest.raises(CognifyFailedError) as exc_info:
        raise_if_cognify_errored({dataset.id: final})
    assert exc_info.value.error_class == "AuthenticationError"
    assert "Incorrect API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_raising_item_path_reraises_root(clean_test_environment, monkeypatch):
    # Default env: the per-item error re-raises, run_tasks yields an enriched
    # errored run info and then re-raises the ROOT exception (which cognify's
    # raise_on_error path wraps into CognifyFailedError).
    monkeypatch.setenv("RAISE_INCREMENTAL_LOADING_ERRORS", "true")

    user, dataset, data_items = await _stage_dataset("situation_one")

    run_infos = []
    with pytest.raises(AuthenticationError):
        async for run_info in run_tasks(
            [Task(failing_task)],
            dataset_id=dataset.id,
            data=data_items,
            user=user,
            pipeline_name="cognify_pipeline",
            incremental_loading=True,
        ):
            run_infos.append(run_info)

    errored = [info for info in run_infos if isinstance(info, PipelineRunErrored)]
    assert errored, f"expected an errored run info before the raise, got {run_infos}"
    assert errored[-1].error_class == "AuthenticationError"


@pytest.mark.asyncio
async def test_typed_cognee_errors_pass_through_cognify_unwrapped(clean_test_environment):
    """`except PermissionDeniedError` around an SDK cognify keeps matching:
    already-typed cognee errors (CogneeApiError subclasses) must not be
    re-wrapped into CognifyFailedError by the run-level except in cognify()."""
    from uuid import uuid4

    from cognee.modules.users.exceptions import PermissionDeniedError

    user = await get_default_user()

    # An unknown dataset UUID raises PermissionDeniedError from dataset
    # authorization inside the pipeline (a name would be auto-created instead).
    with pytest.raises(PermissionDeniedError):
        await cognee.cognify(datasets=[uuid4()], user=user)
