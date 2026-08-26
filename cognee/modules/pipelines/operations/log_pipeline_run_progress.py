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
            # No STARTED row found. In today's only caller (run_tasks.py) this
            # can't happen — log_pipeline_run_start's commit always precedes
            # every item's progress tick, and log_pipeline_run_complete/error
            # only run after all ticks are done — but a future caller could
            # break that ordering, so don't assume it here too.
            terminal_run = (
                await session.execute(
                    select(PipelineRun)
                    .filter(PipelineRun.pipeline_run_id == pipeline_run_id)
                    .filter(
                        PipelineRun.status.in_(
                            [
                                PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                                PipelineRunStatus.DATASET_PROCESSING_ERRORED,
                            ]
                        )
                    )
                    .order_by(PipelineRun.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if terminal_run is not None:
                # The run already finished — inserting a STARTED row now would
                # give it a later created_at than the terminal row, making
                # get_pipeline_status report STARTED for a completed run.
                # Progress is metadata-only, so dropping this tick is harmless.
                return terminal_run

            # Truly no row for this pipeline_run_id yet — e.g. racing
            # log_pipeline_run_start's own commit, which today's only caller
            # (run_tasks.py) can't trigger: it awaits that commit before
            # scheduling any item, so this insert is dead code in practice,
            # kept only for a hypothetical future caller that doesn't. If it
            # ever does fire and log_pipeline_run_start's commit lands right
            # after, the result is two STARTED rows for one pipeline_run_id —
            # not a duplicate update target, since both /status queries pick
            # the latest by created_at regardless. "data" is explicitly None
            # here (log_pipeline_run_start's own summarized-payload field —
            # unavailable to this function, which never receives the raw
            # data) rather than silently omitted, so a row from this path is
            # distinguishable from one log_pipeline_run_start wrote.
            pipeline_run = PipelineRun(
                pipeline_run_id=pipeline_run_id,
                pipeline_name=pipeline_name,
                pipeline_id=pipeline_id,
                status=PipelineRunStatus.DATASET_PROCESSING_STARTED,
                dataset_id=dataset_id,
                run_info={"data": None, "progress": progress},
            )
            session.add(pipeline_run)

        await session.commit()

    return pipeline_run
