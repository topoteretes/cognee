from sqlalchemy import Column, String, Uuid

from cognee.infrastructure.databases.relational import Base


class PipelineRunSession(Base):
    """One session a ``pipeline_runs`` row belongs to; the only place sessions are stored.

    A run launched by an operation bound to several sessions (an ``improve``
    over two sessions) belongs to each of them. Every row that carries
    sessions gets one row here per session, so "the runs of session X" is one
    indexed equality lookup on every relational backend.
    """

    __tablename__ = "pipeline_run_sessions"

    # pipeline_runs.id: the row, not pipeline_run_id (which every row of one
    # run shares).
    pipeline_run_row_id = Column(Uuid, primary_key=True)
    session_id = Column(String, primary_key=True, index=True)
