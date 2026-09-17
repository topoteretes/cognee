from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine

from ..models import PipelineRun
from .get_unclosed_pipeline_runs import TERMINAL_STATUSES


async def pipeline_run_has_terminal_row(pipeline_run_id: UUID) -> bool:
    """True once *pipeline_run_id* has a COMPLETED or ERRORED row of its own.

    Recovery's candidate list is read once, before any dataset lock is taken
    (see ``get_unclosed_pipeline_runs``), so a run that finishes on its own
    between that read and recovery reaching it under the lock is still in
    the stale list. This is the check recovery makes right after acquiring
    the dataset lock, immediately before acting on a candidate: a run that
    holds that same lock releases it only after writing its terminal row
    (``run_pipeline`` holds it across start..complete), so by the time
    recovery gets the lock, a candidate that finished on its own already has
    one. The check can therefore only turn a stale "unclosed" into a fresh
    "closed" — never the other way — so it costs one query and changes
    nothing for a candidate that is genuinely still abandoned.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        query = select(PipelineRun.id).filter(
            PipelineRun.pipeline_run_id == pipeline_run_id,
            PipelineRun.status.in_(TERMINAL_STATUSES),
        )
        return (await session.scalar(query)) is not None
