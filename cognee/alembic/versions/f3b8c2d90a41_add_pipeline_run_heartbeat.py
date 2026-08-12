"""add_pipeline_run_heartbeat

Adds ``pipeline_runs.last_heartbeat_at``, the progress signal used to tell an
in-flight pipeline run apart from an abandoned one.

Existing rows are left NULL rather than backfilled: NULL carries the useful
information "this run never reported progress", and readers fall back to
``created_at`` for those rows, which is exactly the pre-heartbeat behaviour.

Revision ID: f3b8c2d90a41
Revises: e5a7b9c1d3f4
Create Date: 2026-08-12 10:15:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f3b8c2d90a41"
down_revision: Union[str, None] = "e5a7b9c1d3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector, table: str, name: str) -> bool:
    if table not in inspector.get_table_names():
        return False
    return any(column["name"] == name for column in inspector.get_columns(table))


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if not _has_column(inspector, "pipeline_runs", "last_heartbeat_at"):
        op.add_column(
            "pipeline_runs",
            sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if _has_column(inspector, "pipeline_runs", "last_heartbeat_at"):
        op.drop_column("pipeline_runs", "last_heartbeat_at")
