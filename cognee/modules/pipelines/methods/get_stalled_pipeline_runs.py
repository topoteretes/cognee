from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine
from ..models import PipelineRun, PipelineRunStatus


def _last_progress_at(pipeline_run: PipelineRun) -> Optional[datetime]:
    """Most recent evidence the run was alive: heartbeat, else its start time."""
    return pipeline_run.last_heartbeat_at or pipeline_run.created_at


async def get_stalled_pipeline_runs(
    idle_seconds: float = 3600,
    pipeline_name: Optional[str] = None,
    dataset_ids: Optional[list[UUID]] = None,
) -> list[PipelineRun]:
    """Report runs that still claim to be running but have stopped making progress.

    A run qualifies when its latest ``PipelineRun`` row is
    ``DATASET_PROCESSING_STARTED`` and it has not stamped ``last_heartbeat_at``
    within ``idle_seconds``. Because the heartbeat only advances when a pipeline
    task completes, a genuinely slow run -- a local-LLM ingestion that takes
    days -- keeps clearing this filter, while a run whose process died freezes
    and shows up.

    This reports; it does not terminate anything. Callers decide what a stalled
    run means for them (alerting, a dashboard, or recovery as
    ``cognee.modules.cognify.recovery`` does at startup).

    Args:
        idle_seconds: How long a run must have been silent to be reported.
        pipeline_name: Restrict to one pipeline (e.g. ``"cognify_pipeline"``).
        dataset_ids: Restrict to specific datasets.

    Returns:
        The stalled runs' STARTED rows, newest heartbeat first.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        latest_per_dataset = select(
            PipelineRun,
            func.row_number()
            .over(
                partition_by=PipelineRun.dataset_id,
                order_by=PipelineRun.created_at.desc(),
            )
            .label("rn"),
        )

        if pipeline_name is not None:
            latest_per_dataset = latest_per_dataset.filter(
                PipelineRun.pipeline_name == pipeline_name
            )
        if dataset_ids is not None:
            latest_per_dataset = latest_per_dataset.filter(PipelineRun.dataset_id.in_(dataset_ids))

        subquery = latest_per_dataset.subquery()
        latest_run = aliased(PipelineRun, subquery)

        running = (
            (
                await session.execute(
                    select(latest_run)
                    .where(subquery.c.rn == 1)
                    .where(latest_run.status == PipelineRunStatus.DATASET_PROCESSING_STARTED)
                )
            )
            .scalars()
            .all()
        )

    # Staleness is compared in Python rather than SQL: timestamp arithmetic and
    # timezone handling differ between SQLite and Postgres, and the candidate
    # set here is at most one row per dataset.
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=idle_seconds)

    def _is_stalled(pipeline_run: PipelineRun) -> bool:
        progressed_at = _last_progress_at(pipeline_run)
        if progressed_at is None:
            return True
        if progressed_at.tzinfo is None:
            progressed_at = progressed_at.replace(tzinfo=timezone.utc)
        return progressed_at <= cutoff

    stalled = [pipeline_run for pipeline_run in running if _is_stalled(pipeline_run)]

    return sorted(
        stalled,
        key=lambda run: _last_progress_at(run) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
