"""E2E contract test: an exception raised inside a pipeline task is never swallowed.

Runs a real cognee pipeline (memify with custom tasks over an added dataset)
whose enrichment task raises, and asserts the three parts of the error
contract:

1. The ORIGINAL exception (type and message) propagates out of the blocking
   SDK call to the caller — not a generic wrapper, not a silent return.
2. The pipeline run is recorded as DATASET_PROCESSING_ERRORED in the
   relational DB, so status APIs report the failure.
3. The system stays usable afterwards: a healthy pipeline run on the same
   dataset completes — one failed run must not wedge engines or state.

Uses only custom passthrough tasks, so no LLM or embedding calls are made.
"""

import os
import pathlib

import cognee
from cognee.modules.data.methods import get_datasets_by_name
from cognee.modules.pipelines.operations.get_pipeline_status import get_pipeline_status
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

logger = get_logger()

DATASET_NAME = "pipeline_error_propagation"


class KaboomError(Exception):
    """Marker exception: only the failing task raises it, so catching exactly
    this type proves the original exception object survived the pipeline."""


async def passthrough_extraction(data):
    return data


async def exploding_enrichment(data):
    raise KaboomError("task exploded on purpose")


async def healthy_enrichment(data):
    return data


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_pipeline_error")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_pipeline_error")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add("Pipeline error propagation seed text.", dataset_name=DATASET_NAME)

    # 1. The task's own exception must reach the caller.
    raised = None
    try:
        await cognee.memify(
            extraction_tasks=[Task(passthrough_extraction)],
            enrichment_tasks=[Task(exploding_enrichment)],
            data=["payload for the failing pipeline"],
            dataset=DATASET_NAME,
        )
    except KaboomError as error:
        raised = error
    assert raised is not None, "memify swallowed the task exception instead of raising it"
    assert "task exploded on purpose" in str(raised), (
        f"original error message was lost; got: {raised!r}"
    )
    logger.info("Task exception propagated to the caller as expected")

    # 2. The failed run must be recorded as errored in the relational DB.
    user = await get_default_user()
    datasets = await get_datasets_by_name(DATASET_NAME, user.id)
    assert datasets, f"dataset {DATASET_NAME!r} not found after add()"
    dataset_id = datasets[0].id

    statuses = await get_pipeline_status([dataset_id], "memify_pipeline")
    status = statuses.get(str(dataset_id)) or statuses.get(dataset_id)
    assert status is not None, f"no memify_pipeline run recorded for dataset {dataset_id}"
    recorded_status = getattr(status, "status", status)
    assert str(recorded_status) == str(PipelineRunStatus.DATASET_PROCESSING_ERRORED), (
        f"pipeline run not recorded as errored; got: {recorded_status!r}"
    )
    logger.info("Failed pipeline run recorded as DATASET_PROCESSING_ERRORED")

    # 3. One failed run must not wedge the system: a healthy run now succeeds.
    result = await cognee.memify(
        extraction_tasks=[Task(passthrough_extraction)],
        enrichment_tasks=[Task(healthy_enrichment)],
        data=["payload for the healthy pipeline"],
        dataset=DATASET_NAME,
    )
    assert result, "healthy pipeline run after a failed one returned no run info"

    statuses = await get_pipeline_status([dataset_id], "memify_pipeline")
    status = statuses.get(str(dataset_id)) or statuses.get(dataset_id)
    recorded_status = getattr(status, "status", status)
    assert str(recorded_status) == str(PipelineRunStatus.DATASET_PROCESSING_COMPLETED), (
        f"healthy run after the failed one did not complete; got: {recorded_status!r}"
    )
    logger.info("Healthy pipeline run completed after the failed one")

    print("Pipeline error propagation test passed")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
