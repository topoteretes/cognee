from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import aliased

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import PipelineRun, PipelineRunStatus

_TERMINAL_STATUSES = (
    PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
    PipelineRunStatus.DATASET_PROCESSING_ERRORED,
)


async def get_unclosed_pipeline_runs(
    dataset_ids: list[UUID] | None = None,
) -> list[PipelineRun]:
    """Every STARTED pipeline run that never got a terminal row of its own.

    Deliberately not built on the latest-run-per-dataset lookups next door.
    Those answer "what is this dataset's current status", where only the
    newest row matters; startup recovery has to find *every* run a crash left
    unclosed. One crash can abandon several runs of the same dataset and
    pipeline (a background batch writes all its STARTED rows up front), and
    ranking by recency would hide all but the newest of them.

    A run is unclosed when a ``DATASET_PROCESSING_STARTED`` row exists for its
    ``pipeline_run_id`` and no ``COMPLETED`` or ``ERRORED`` row does. Runs are
    returned oldest first, one row per run: ``log_pipeline_run_progress`` can
    insert a second STARTED row for the same run, and a caller closing a run
    wants to see it once.

    Operation records (``record_operation``) carry no ``pipeline_name`` and no
    status, so they are excluded rather than mistaken for runs.

    dataset_ids=None covers every dataset; pass a list (possibly empty) to
    scope it.
    """
    if dataset_ids is not None and not dataset_ids:
        return []

    # Aliased so the NOT IN subquery keeps its own FROM: sharing the outer
    # table would let SQLAlchemy correlate it and silently change the test to
    # "this row is not itself terminal".
    closed = aliased(PipelineRun)
    closed_run_ids = select(closed.pipeline_run_id).filter(closed.status.in_(_TERMINAL_STATUSES))

    query = select(PipelineRun).filter(
        PipelineRun.status == PipelineRunStatus.DATASET_PROCESSING_STARTED,
        PipelineRun.pipeline_name.isnot(None),
        PipelineRun.pipeline_run_id.notin_(closed_run_ids),
    )
    if dataset_ids is not None:
        query = query.filter(PipelineRun.dataset_id.in_(dataset_ids))

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        rows = (
            (await session.execute(query.order_by(PipelineRun.created_at, PipelineRun.id)))
            .scalars()
            .all()
        )

    unclosed: dict[UUID, PipelineRun] = {}
    for row in rows:
        unclosed.setdefault(row.pipeline_run_id, row)

    return list(unclosed.values())
