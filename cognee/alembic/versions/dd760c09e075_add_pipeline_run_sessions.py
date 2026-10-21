"""Add pipeline_run_sessions (CLO-1105)

Revision ID: dd760c09e075
Revises: d4a8e2f6b1c9
Create Date: 2026-10-21 15:54:34.739646

One row per (pipeline_runs row, session). A run launched by an operation bound
to several sessions (an improve over two sessions) belongs to each of them,
which the single pipeline_runs.session_id column cannot hold; that column keeps
the session when there is exactly one. Filtering "the runs of session X" is
then an indexed equality lookup on every relational backend, with no JSON
membership query.

Backfill: every existing pipeline_runs row with a session_id gets its row here,
so runs stamped before this revision are still found by session.

Idempotent: the table is created, and backfilled, only when it is missing.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd760c09e075"
down_revision: str | None = "d4a8e2f6b1c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "pipeline_run_sessions"
INDEX_NAME = "ix_pipeline_run_sessions_session_id"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(TABLE_NAME):
        return
    op.create_table(
        TABLE_NAME,
        sa.Column("pipeline_run_row_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("pipeline_run_row_id", "session_id"),
    )
    op.create_index(INDEX_NAME, TABLE_NAME, ["session_id"], unique=False)

    if inspector.has_table("pipeline_runs"):
        op.execute(
            f"INSERT INTO {TABLE_NAME} (pipeline_run_row_id, session_id) "
            "SELECT id, session_id FROM pipeline_runs WHERE session_id IS NOT NULL"
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(TABLE_NAME):
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
        op.drop_table(TABLE_NAME)
