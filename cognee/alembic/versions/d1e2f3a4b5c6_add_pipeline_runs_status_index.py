"""Add composite index on pipeline_runs for the /status latest-run lookup

Revision ID: d1e2f3a4b5c6
Revises: b3d5f7a9c1e2
Create Date: 2026-08-17

get_pipeline_status.py / get_pipeline_progress.py both run a ROW_NUMBER()
window query filtered by (dataset_id, pipeline_name) and ordered by
created_at DESC to find each dataset's latest PipelineRun. dataset_id
already has its own single-column index, but the filter+sort as a whole
was not covered by one index — and CLO-557's in-flight progress ticks
(log_pipeline_run_progress) add rows to this table throughout a run, not
just at start/complete/error, making that gap matter sooner than it used to.
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "b3d5f7a9c1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_pipeline_runs_dataset_pipeline_created_at"


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("pipeline_runs")]
    if INDEX_NAME not in existing_indexes:
        if conn.dialect.name == "postgresql":
            # CREATE INDEX (without CONCURRENTLY) takes a table-wide lock for
            # the duration of the build — fine for the small dev/test tables
            # every other migration in this repo targets, but pipeline_runs
            # now grows throughout a run (see module docstring), so a large
            # production table deserves the non-blocking path. CONCURRENTLY
            # cannot run inside a transaction, hence autocommit_block().
            #
            # IF NOT EXISTS matters here specifically: if a prior CONCURRENTLY
            # build died mid-way (deadlock/timeout/crash), Postgres leaves an
            # INVALID index under this name. The `existing_indexes` guard
            # above only reflects valid indexes, so a retry would otherwise
            # hit "relation already exists" instead of cleanly no-op'ing or
            # replacing the invalid one.
            with op.get_context().autocommit_block():
                op.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON pipeline_runs "
                    f"(dataset_id, pipeline_name, created_at)"
                )
        else:
            op.create_index(
                INDEX_NAME,
                "pipeline_runs",
                ["dataset_id", "pipeline_name", "created_at"],
            )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("pipeline_runs")]
    if INDEX_NAME in existing_indexes:
        if conn.dialect.name == "postgresql":
            with op.get_context().autocommit_block():
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        else:
            op.drop_index(INDEX_NAME, "pipeline_runs")
