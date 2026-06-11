"""Add the single-row global_database_version table

Tracks the deployment-wide Cognee version (written on every startup in both
access-control modes) and, when backend access control is disabled, the
data-migration revision of the GLOBAL graph/vector databases (no per-dataset
dataset_database rows exist to carry them in that mode).

Revision ID: d8f4a1b2c3e9
Revises: c1a2b3d4e5f9
Create Date: 2026-06-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d8f4a1b2c3e9"
down_revision: Union[str, None] = "c1a2b3d4e5f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "global_database_version"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if not inspector.has_table(TABLE_NAME):
        op.create_table(
            TABLE_NAME,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("cognee_version", sa.String(), nullable=True),
            sa.Column("global_migration_revision", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if inspector.has_table(TABLE_NAME):
        op.drop_table(TABLE_NAME)
