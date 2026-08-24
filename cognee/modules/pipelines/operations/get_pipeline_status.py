from uuid import UUID

from ..methods import get_latest_pipeline_runs_by_datasets


async def get_pipeline_status(dataset_ids: list[UUID], pipeline_name: str):
    runs = await get_latest_pipeline_runs_by_datasets(dataset_ids, pipeline_name)

    return {str(dataset_id): run.status for dataset_id, run in runs.items()}
