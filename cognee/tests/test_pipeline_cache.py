"""
Test suite for the pipeline_cache feature in Cognee pipelines.

This module tests the behavior of the `pipeline_cache` parameter which controls
whether a pipeline should skip re-execution when it has already been completed
for the same dataset.

Architecture Overview:
---------------------
The pipeline_cache mechanism works at the dataset level:
1. When a pipeline runs, it logs its status (INITIATED -> STARTED -> COMPLETED)
2. Before each run, `check_pipeline_run_qualification()` checks the pipeline status
3. If `use_pipeline_cache=True` and status is COMPLETED/STARTED, the pipeline skips
4. If `use_pipeline_cache=False`, the pipeline always re-executes regardless of status
"""

import pytest

import cognee
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.pipelines import run_pipeline
from cognee.modules.users.methods import get_default_user

from cognee.modules.pipelines.layers.reset_dataset_pipeline_run_status import (
    reset_dataset_pipeline_run_status,
)
from cognee.infrastructure.databases.relational import create_db_and_tables


class ExecutionCounter:
    """Helper class to track task execution counts."""

    def __init__(self):
        self.count = 0


async def create_counting_task(data, counter: ExecutionCounter):
    """Create a task that increments a counter from the ExecutionCounter instance when executed."""
    counter.count += 1
    return counter


class TestPipelineCache:
    """Tests for basic pipeline_cache on/off behavior."""

    @pytest.mark.asyncio
    async def test_pipeline_cache_off_allows_reexecution(self):
        """
        Test that with use_pipeline_cache=False, the pipeline re-executes
        even when it has already completed for the dataset.

        Expected behavior:
        - First run: Pipeline executes fully, task runs once
        - Second run: Pipeline executes again, task runs again (total: 2 times)
        """
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await create_db_and_tables()

        counter = ExecutionCounter()
        user = await get_default_user()

        tasks = [Task(create_counting_task, counter=counter)]

        # First run
        pipeline_results_1 = []
        async for result in run_pipeline(
            tasks=tasks,
            datasets="test_dataset_cache_off",
            data=["sample data"],  # Data is necessary to trigger processing
            user=user,
            pipeline_name="test_cache_off_pipeline",
            use_pipeline_cache=False,
        ):
            pipeline_results_1.append(result)

        first_run_count = counter.count
        assert first_run_count >= 1, "Task should have executed at least once on first run"

        # Second run with pipeline_cache=False
        pipeline_results_2 = []
        async for result in run_pipeline(
            tasks=tasks,
            datasets="test_dataset_cache_off",
            data=["sample data"],  # Data is necessary to trigger processing
            user=user,
            pipeline_name="test_cache_off_pipeline",
            use_pipeline_cache=False,
        ):
            pipeline_results_2.append(result)

        second_run_count = counter.count
        assert second_run_count > first_run_count, (
            f"With pipeline_cache=False, task should re-execute. "
            f"First run: {first_run_count}, After second run: {second_run_count}"
        )

    @pytest.mark.asyncio
    async def test_reset_pipeline_status_allows_reexecution_with_cache(self):
        """
        Test that resetting pipeline status allows re-execution even with
        pipeline_cache=True.
        """
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await create_db_and_tables()

        counter = ExecutionCounter()
        user = await get_default_user()
        dataset_name = "reset_status_test"
        pipeline_name = "test_reset_pipeline"

        tasks = [Task(create_counting_task, counter=counter)]

        # First run
        pipeline_result = []
        async for result in run_pipeline(
            tasks=tasks,
            datasets=dataset_name,
            user=user,
            data=["sample data"],  # Data is necessary to trigger processing
            pipeline_name=pipeline_name,
            use_pipeline_cache=True,
        ):
            pipeline_result.append(result)

        first_run_count = counter.count
        assert first_run_count >= 1

        # Second run without reset - should skip
        async for _ in run_pipeline(
            tasks=tasks,
            datasets=dataset_name,
            user=user,
            data=["sample data"],  # Data is necessary to trigger processing
            pipeline_name=pipeline_name,
            use_pipeline_cache=True,
        ):
            pass

        after_second_run = counter.count
        assert after_second_run == first_run_count, "Should have skipped due to cache"

        # Reset the pipeline status
        await reset_dataset_pipeline_run_status(
            pipeline_result[0].dataset_id, user, pipeline_names=[pipeline_name]
        )

        # Third run after reset - should execute
        async for _ in run_pipeline(
            tasks=tasks,
            datasets=dataset_name,
            user=user,
            data=["sample data"],  # Data is necessary to trigger processing
            pipeline_name=pipeline_name,
            use_pipeline_cache=True,
        ):
            pass

        after_reset_run = counter.count
        assert after_reset_run > after_second_run, (
            f"After reset, pipeline should re-execute. "
            f"Before reset: {after_second_run}, After reset run: {after_reset_run}"
        )
