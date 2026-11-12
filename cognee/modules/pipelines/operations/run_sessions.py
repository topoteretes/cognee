"""Store and read the sessions a ``pipeline_runs`` row belongs to (``pipeline_run_sessions``)."""

from collections.abc import Iterable, Sequence
from uuid import UUID, uuid4

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.pipelines.models.PipelineRun import PipelineRun
from cognee.modules.pipelines.models.PipelineRunSession import PipelineRunSession


def add_run_sessions(db_session, pipeline_run: PipelineRun, session_ids: Sequence[str]) -> None:
    """Add ``pipeline_run`` and one ``pipeline_run_sessions`` row per session to ``db_session``.

    The caller commits, so the run row and its sessions land in one
    transaction. The row id is assigned here because the session rows need
    it before the flush that would otherwise assign it.
    """
    if pipeline_run.id is None:
        pipeline_run.id = uuid4()
    db_session.add(pipeline_run)
    for session_id in session_ids:
        db_session.add(
            PipelineRunSession(pipeline_run_row_id=pipeline_run.id, session_id=session_id)
        )


async def get_run_session_ids(row_ids: Iterable[UUID]) -> dict[UUID, list[str]]:
    """The sessions of each of these ``pipeline_runs`` rows, by row id, sorted."""
    row_ids = list(row_ids)
    if not row_ids:
        return {}
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as db_session:
        rows = (
            await db_session.execute(
                select(PipelineRunSession.pipeline_run_row_id, PipelineRunSession.session_id).where(
                    PipelineRunSession.pipeline_run_row_id.in_(row_ids)
                )
            )
        ).all()
    sessions: dict[UUID, list[str]] = {}
    for row_id, session_id in rows:
        sessions.setdefault(row_id, []).append(session_id)
    return {row_id: sorted(ids) for row_id, ids in sessions.items()}


def rows_of_session(session_id: str):
    """A subquery of the ``pipeline_runs`` row ids that belong to ``session_id``."""
    return select(PipelineRunSession.pipeline_run_row_id).where(
        PipelineRunSession.session_id == session_id
    )
