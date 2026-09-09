from uuid import UUID

from ..methods import get_effective_pipeline_status, get_latest_pipeline_runs_by_datasets


async def get_pipeline_status(dataset_ids: list[UUID], pipeline_name: str):
    # Raw/control-flow variant: returns the stored PipelineRunStatus enum
    # members untouched. check_pipeline_run_qualification compares the
    # result against PipelineRunStatus.DATASET_PROCESSING_STARTED /
    # _COMPLETED to decide whether a pipeline is already running or already
    # done — do not apply get_effective_pipeline_status() here, that would
    # make both comparisons permanently False. The reporting sibling is
    # get_effective_pipeline_status_by_datasets, below.
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {str(dataset_id): run.status for dataset_id, run in runs.items()}


async def get_pipeline_progress(dataset_ids: list[UUID], pipeline_name: str):
    """Same latest-run lookup as get_pipeline_status, plus the in-flight
    progress snapshot (see log_pipeline_run_progress). A separate function —
    and a separate /status/progress endpoint — rather than a flag on
    get_pipeline_status/get_status, so neither call's response shape ever
    branches at runtime on how it was called.

    run_info["progress"] is written incrementally as items/stages complete
    (see run_tasks.py / run_tasks_data_item.py); it is only present while a
    run is STARTED, so absence just means "no progress ticks yet" rather
    than an error.

    Raw/control-flow variant, same caveat as get_pipeline_status: this
    returns the stored PipelineRunStatus, not the effective one. The
    reporting sibling is get_effective_pipeline_progress_by_datasets, below.
    """
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {
        str(dataset_id): {
            "status": run.status,
            "progress": (run.run_info or {}).get("progress"),
        }
        for dataset_id, run in runs.items()
    }


async def get_effective_pipeline_status_by_datasets(dataset_ids: list[UUID], pipeline_name: str):
    """Reporting sibling of get_pipeline_status: same latest-run lookup, but
    each value is the read-time EffectivePipelineRunStatus (a stale STARTED
    row reads as ABANDONED) instead of the raw stored PipelineRunStatus.

    For display only — GET /v1/datasets/status uses this (the activity feed
    calls get_effective_pipeline_status per row directly instead, since it
    already has the run object from its own query). Anything that decides
    whether to launch a pipeline must use get_pipeline_status instead; see
    its comment for why.
    """
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {
        str(dataset_id): get_effective_pipeline_status(
            run,
            # get_latest_pipeline_runs_by_datasets returns each dataset's
            # single newest row. A STARTED row of a run that already has a
            # terminal row can never be that newest row —
            # log_pipeline_run_progress drops progress ticks instead of
            # inserting a later STARTED row once a terminal row exists for
            # the same pipeline_run_id — so a STARTED row surfacing here
            # never has a terminal sibling.
            run_has_terminal_row=False,
        )
        for dataset_id, run in runs.items()
    }


async def get_effective_pipeline_progress_by_datasets(dataset_ids: list[UUID], pipeline_name: str):
    """Reporting sibling of get_pipeline_progress: same {status, progress}
    shape, but status is the read-time EffectivePipelineRunStatus rather
    than the raw stored PipelineRunStatus. Backs GET /v1/datasets/status/progress.
    """
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {
        str(dataset_id): {
            # Same reasoning as get_effective_pipeline_status_by_datasets:
            # the newest row per dataset can never be a STARTED row with a
            # terminal sibling.
            "status": get_effective_pipeline_status(run, run_has_terminal_row=False),
            "progress": (run.run_info or {}).get("progress"),
        }
        for dataset_id, run in runs.items()
    }
