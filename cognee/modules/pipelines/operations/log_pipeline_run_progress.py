from uuid import UUID
from typing import Optional
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


async def log_pipeline_run_progress(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    completed_items: int,
    total_items: int,
    current_stage: Optional[str] = None,
):
    # Progress is metadata within the STARTED state, not a new PipelineRunStatus —
    # get_pipeline_status.py always reads the latest row per dataset, so a fresh
    # row here (matching the insert-per-event pattern of log_pipeline_run_start/
    # log_pipeline_run_complete) is picked up without any other query changes.
    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
        dataset_id=dataset_id,
        run_info={
            "progress": {
                "completed_items": completed_items,
                "total_items": total_items,
                "current_stage": current_stage,
            },
        },
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
