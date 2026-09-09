from uuid import UUID

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus


async def log_pipeline_run_initiated(
    pipeline_run_id: UUID,
    pipeline_id: UUID,
    pipeline_name: str,
    dataset_id: UUID,
    *,
    user_id: UUID | None = None,
):
    """Mark a pipeline as eligible to run again by appending an INITIATED row.

    The id belongs to the run being superseded, it is not a new one. Minting a
    fresh id here (which this writer used to do) created a run identity that no
    STARTED or terminal row ever shared, so the row sat in ``pipeline_runs``
    forever describing work that had already finished. The row is control
    state rather than history: ``log_pipeline_run_start`` clears it when the
    run it unblocks actually begins.
    """
    pipeline_run = PipelineRun(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        pipeline_id=pipeline_id,
        status=PipelineRunStatus.DATASET_PROCESSING_INITIATED,
        dataset_id=dataset_id,
        run_info={},
        user_id=user_id,
        operation_name=pipeline_name,
    )

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        session.add(pipeline_run)
        await session.commit()

    return pipeline_run
