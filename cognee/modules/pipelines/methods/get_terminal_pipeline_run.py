from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

_TERMINAL_STATUSES = (
    PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    PipelineRunStatus.DATASET_PROCESSING_ERRORED,
)


async def get_terminal_pipeline_run(pipeline_run_id: UUID) -> PipelineRun | None:
    """The existing terminal row for this run, if one was already written.

    ``pipeline_runs`` is append-only: INITIATED, STARTED, and a terminal row
    (COMPLETED or ERRORED) each insert rather than update, all sharing one
    ``pipeline_run_id``. Two terminal writers can otherwise both fire for the
    same run — startup recovery closing a run as ERRORED while the process it
    thought was dead is still running and later calls
    ``log_pipeline_run_complete`` itself — leaving COMPLETED written over a
    graph the recovery rollback already deleted. Checking first and skipping
    the second write turns that into "the run keeps its first terminal
    status" instead of a COMPLETED row that lied about what happened to the
    graph.

    This is a check-then-insert, not a lock: two terminal writes racing at
    the database level within the same instant can still both land. It closes
    the sequential case, which is what the startup-recovery race actually is.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = (
            select(PipelineRun)
            .filter(
                PipelineRun.pipeline_run_id == pipeline_run_id,
                PipelineRun.status.in_(_TERMINAL_STATUSES),
            )
            .order_by(PipelineRun.created_at.desc())
        )
        return (await session.execute(query)).scalars().first()
