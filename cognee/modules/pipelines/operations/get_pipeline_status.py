from uuid import UUID

from ..methods import get_latest_pipeline_runs_by_datasets


async def get_pipeline_status(dataset_ids: list[UUID], pipeline_name: str):
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
    """
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {
        str(dataset_id): {
            "status": run.status,
            "progress": (run.run_info or {}).get("progress"),
        }
        for dataset_id, run in runs.items()
    }
