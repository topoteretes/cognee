"""Functional test for "situation 2": a per-item task failure must not be silent.

Drives the REAL run_tasks pipeline (relational DB, dataset row, run records)
with a task that raises per item. Before COG-6276 this path swallowed the
cause: the yielded run info and the pipeline_runs row said only "Pipeline run
failed. Data item could not be processed." — the silent first-build failure
that 854/854 churned accounts hit. Asserts the ROOT cause now survives to:

1. the yielded ``PipelineRunErrored`` (error_class / scrubbed error_message),
2. the persisted ``pipeline_runs`` row,
3. the typed ``CognifyFailedError`` cognify raises from that run info.
"""

import asyncio
import os

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


async def run_situation_two():
    # The silent path: per-item errors are collected instead of re-raised.
    # (With the default "true", the error re-raises out of the item worker
    # and takes the run-level path instead — that variant is covered below.)
    os.environ["RAISE_INCREMENTAL_LOADING_ERRORS"] = "false"

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await create_db_and_tables()
    user = await get_default_user()

    # add() creates the dataset + data rows without any LLM call.
    await cognee.add("situation two probe text", dataset_name="situation_two", user=user)

    from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data

    [dataset] = await get_authorized_existing_datasets(["situation_two"], "read", user)
    data_items = await get_dataset_data(dataset.id)
    assert data_items, "add() should have created a data row"

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
    try:
        from cognee.api.v1.cognify.cognify import raise_if_cognify_errored

        raise_if_cognify_errored({dataset.id: final})
        raise AssertionError("raise_if_cognify_errored did not raise")
    except CognifyFailedError as error:
        assert error.error_class == "AuthenticationError"
        assert "Incorrect API key" in str(error)


async def run_situation_one():
    # Default env: the per-item error re-raises, run_tasks yields an enriched
    # errored run info and then re-raises the ROOT exception (which cognify's
    # raise_on_error path wraps into CognifyFailedError).
    os.environ["RAISE_INCREMENTAL_LOADING_ERRORS"] = "true"

    user = await get_default_user()
    from cognee.modules.data.methods import get_authorized_existing_datasets, get_dataset_data

    [dataset] = await get_authorized_existing_datasets(["situation_two"], "read", user)
    data_items = await get_dataset_data(dataset.id)

    run_infos = []
    try:
        async for run_info in run_tasks(
            [Task(failing_task)],
            dataset_id=dataset.id,
            data=data_items,
            user=user,
            pipeline_name="cognify_pipeline",
            incremental_loading=True,
        ):
            run_infos.append(run_info)
        raise AssertionError("expected the root exception to re-raise")
    except AuthenticationError:
        pass

    errored = [info for info in run_infos if isinstance(info, PipelineRunErrored)]
    assert errored, f"expected an errored run info before the raise, got {run_infos}"
    assert errored[-1].error_class == "AuthenticationError"


def test_per_item_failure_propagates_root_cause():
    asyncio.run(run_situation_two())
    asyncio.run(run_situation_one())


if __name__ == "__main__":
    test_per_item_failure_propagates_root_cause()
