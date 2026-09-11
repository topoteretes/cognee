from uuid import UUID

from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.operations.log_pipeline_run_initiated import (
    log_pipeline_run_initiated,
)


async def reset_pipeline_run_status(pipeline_run: PipelineRun, user_id: UUID):
    """Clear a finished or stuck status so the pipeline may run again.

    Takes the run being reset rather than its parts. The marker describes that
    run, so its identity comes from the run's own row: re-deriving it from the
    acting user (what this used to do) stamped a different pipeline_id whenever
    the resetting caller was not the user who ran the pipeline, which startup
    recovery always is, since it acts as the dataset owner.

    *user_id* is who asked for the reset, which is not necessarily the run's own
    user, and is recorded as such. It is required rather than defaulted: a
    marker with a NULL user_id is one /activity/pipeline-runs can show nobody,
    since it filters visibility on that column.
    """
    # Without this the pipeline status will be DATASET_PROCESSING_COMPLETED and will skip the execution.
    await log_pipeline_run_initiated(
        pipeline_run_id=pipeline_run.pipeline_run_id,
        pipeline_id=pipeline_run.pipeline_id,
        pipeline_name=pipeline_run.pipeline_name,
        dataset_id=pipeline_run.dataset_id,
        user_id=user_id,
    )
