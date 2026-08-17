from uuid import UUID
from typing import Optional
from sqlalchemy import select
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
    """Persist an in-flight progress snapshot for this run.

    Progress is metadata within the STARTED state, not a new PipelineRunStatus.
    Unlike log_pipeline_run_start/complete/error — which each insert a new
    terminal-or-initial row by design — this UPDATES the run's existing
    STARTED row in place. A run can tick many times (throttled to ~20 per
    run in run_tasks.py, but still N per batch); inserting a fresh row per
    tick would grow pipeline_runs without bound as batches accumulate over
    the table's lifetime, not just within one run.

    Concurrent ticks for the same run may race (read-modify-write on the
    same row from different sessions) — last write wins, no locking. That's
    fine for a display-only "how far along are we" signal; it only risks a
    slightly stale snapshot between ticks, never a growing table.
    """
    db_engine = get_relational_engine()

    progress = {
        "completed_items": completed_items,
        "total_items": total_items,
        "current_stage": current_stage,
    }

    async with db_engine.get_async_session() as session:
        pipeline_run = (
            await session.execute(
                select(PipelineRun)
                .filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                .filter(PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_STARTED)
                .order_by(PipelineRun.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        if pipeline_run is not None:
            run_info = dict(pipeline_run.run_info or {})
            run_info["progress"] = progress
            pipeline_run.run_info = run_info
            session.add(pipeline_run)
        else:
            # No STARTED row found — e.g. called racing log_pipeline_run_start's
            # own commit, or after the run already reached a terminal status.
            # Insert rather than drop the tick, matching the pre-existing
            # log_pipeline_run_start/complete/error insert pattern.
            pipeline_run = PipelineRun(
                pipeline_run_id=pipeline_run_id,
                pipeline_name=pipeline_name,
                pipeline_id=pipeline_id,
                status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                dataset_id=dataset_id,
                run_info={"progress": progress},
            )
            session.add(pipeline_run)

        await session.commit()

    return pipeline_run
