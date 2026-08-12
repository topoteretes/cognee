"""add_pipeline_run_ownership

Adds ``pipeline_runs.owner_node_id`` and ``pipeline_runs.owner_pid``, which let
startup recovery ask the operating system whether a run's process is still
running instead of inferring liveness from the age of its heartbeat.

Existing rows are left NULL. A run with no recorded owner cannot be checked, so
readers fall back to the heartbeat, which is the pre-ownership behaviour.

Revision ID: a4d7e2f10c85
Revises: f3b8c2d90a41
Create Date: 2026-08-12 12:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4d7e2f10c85"
down_revision: Union[str, None] = "f3b8c2d90a41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = {
    "owner_node_id": sa.String(),
    "owner_pid": sa.Integer(),
}


def _has_column(inspector, table: str, name: str) -> bool:
    if table not in inspector.get_table_names():
        return False
    return any(column["name"] == name for column in inspector.get_columns(table))


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    for name, column_type in _COLUMNS.items():
        if not _has_column(inspector, "pipeline_runs", name):
            op.add_column("pipeline_runs", sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    for name in _COLUMNS:
        if _has_column(inspector, "pipeline_runs", name):
            op.drop_column("pipeline_runs", name)
