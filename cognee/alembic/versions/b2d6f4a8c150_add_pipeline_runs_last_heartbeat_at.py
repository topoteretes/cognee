"""Add last_heartbeat_at to pipeline_runs so staleness stops being a guess

Revision ID: b2d6f4a8c150
Revises: a7c2e9f4b8d1
Create Date: 2026-09-10

PipelineRun gained a nullable ``last_heartbeat_at`` column
(cognee/modules/pipelines/models/PipelineRun.py) recording when a run last
showed a sign of life, so readers can tell a run that is still working from
one whose process died instead of guessing from the row's age.

Deliberately no backfill. Existing rows keep NULL, and readers fall back to
``created_at`` for them, which is exactly today's behaviour — so this
migration changes nothing on its own. Backfilling ``created_at`` into the
column would be indistinguishable from a genuine tick and would claim
evidence the run never produced.

No index: the column is only ever read alongside a row already selected by
the existing dataset_id/pipeline_name/created_at index, and this row is
updated repeatedly while a run is in flight, where a second index would cost
write throughput on the default SQLite backend for no read benefit.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2d6f4a8c150"
down_revision: str | None = "a7c2e9f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "pipeline_runs"
COLUMN_NAME = "last_heartbeat_at"


def _has_column(inspector, table: str, name: str) -> bool:
    return any(col["name"] == name for col in inspector.get_columns(table))


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if not _has_column(insp, TABLE_NAME, COLUMN_NAME):
        op.add_column(
            TABLE_NAME,
            sa.Column(COLUMN_NAME, sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if _has_column(insp, TABLE_NAME, COLUMN_NAME):
        # batch mode: SQLite cannot DROP COLUMN in place.
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.drop_column(COLUMN_NAME)
