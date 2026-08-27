import os
from datetime import datetime, timedelta, timezone
from uuid import UUID
from typing import Optional, List

from cognee.modules.pipelines.methods import get_pipeline_runs_by_dataset, reset_pipeline_run_status
from cognee.modules.pipelines.models.PipelineRun import PipelineRunStatus
from cognee.modules.users.models import User

STALE_INITIATED_THRESHOLD_SECONDS = int(
    os.getenv("COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS", "3600")
)


def _is_stale_initiated(created_at) -> bool:
    if created_at is None:
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STALE_INITIATED_THRESHOLD_SECONDS)
    return created_at <= cutoff


async def reset_dataset_pipeline_run_status(
    dataset_id: UUID, user: User, pipeline_names: Optional[list[str]] = None
):
    """Reset the status of all (or selected) pipeline runs for a dataset.

    If *pipeline_names* is given, only runs whose *pipeline_name* is in
    that list are touched.
    """
    related_pipeline_runs = await get_pipeline_runs_by_dataset(dataset_id)

    for pipeline_run in related_pipeline_runs:
        # INITIATED is normally left alone so a freshly created run is not
        # clobbered, but a stale INITIATED (no STARTED within threshold) means
        # the pipeline never made progress — e.g. queue deadlock / crash before
        # log_pipeline_run_start. Those must be recoverable or datasets stay
        # stuck in DATASET_PROCESSING_INITIATED forever (observed 56h outage).
        if pipeline_run.status is PipelineRunStatus.DATASET_PROCESSING_INITIATED:
            if not _is_stale_initiated(getattr(pipeline_run, "created_at", None)):
                continue

        # If a name filter is provided, skip non-matching runs
        if pipeline_names is not None and pipeline_run.pipeline_name not in pipeline_names:
            continue

        await reset_pipeline_run_status(user.id, dataset_id, pipeline_run.pipeline_name)
