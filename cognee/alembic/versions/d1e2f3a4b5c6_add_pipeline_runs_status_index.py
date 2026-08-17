"""Add composite index on pipeline_runs for the /status latest-run lookup

Revision ID: d1e2f3a4b5c6
Revises: b8c1d3e5f7a9
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
down_revision: Union[str, None] = "b8c1d3e5f7a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "ix_pipeline_runs_dataset_pipeline_created_at"


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("pipeline_runs")]
    if INDEX_NAME not in existing_indexes:
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
        op.drop_index(INDEX_NAME, "pipeline_runs")
