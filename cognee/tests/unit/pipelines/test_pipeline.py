from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.pipelines.operations.pipeline import (
    run_pipeline_per_dataset,
)

from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunStarted,
    PipelineRunCompleted,
)

DATASET_ID = uuid4()

class FakeDataset:
    id = DATASET_ID
    name = "test_dataset"

class FakeUser:
    id = uuid4()

async def fake_run_tasks(*args, **kwargs):
    yield PipelineRunStarted(
        pipeline_run_id=uuid4(),
        dataset_id=DATASET_ID,
        dataset_name="test_dataset",
        payload=[],
    )

    yield PipelineRunCompleted(
        pipeline_run_id=uuid4(),
        dataset_id=DATASET_ID,
        dataset_name="test_dataset",
    )

@pytest.mark.asyncio
async def test_pipeline_emits_single_start_event_when_cache_disabled():

    with (
        patch(
            "cognee.modules.pipelines.operations.pipeline.get_dataset_data",
            AsyncMock(return_value=["dummy"]),
        ),
        patch(
            "cognee.modules.pipelines.operations.pipeline.check_pipeline_run_qualification",
            AsyncMock(
                return_value=PipelineRunStarted(
                    pipeline_run_id=uuid4(),
                    dataset_id=DATASET_ID,
                    dataset_name="test_dataset",
                    payload=[],
                )
            ),
        ),
        patch(
            "cognee.modules.pipelines.operations.pipeline.run_tasks",
            fake_run_tasks,
        ),
    ):

        events = []

        async for event in run_pipeline_per_dataset(
            dataset=FakeDataset(),
            user=FakeUser(),
            tasks=[],
            data=[],
            use_pipeline_cache=False,
        ):
            events.append(type(event).__name__)

        assert events.count("PipelineRunStarted") == 1
        assert events == [
            "PipelineRunStarted",
            "PipelineRunCompleted",
        ]