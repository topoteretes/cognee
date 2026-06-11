"""Add Cognee version and migration revision tracking to dataset_database

Revision ID: c1a2b3d4e5f9
Revises: 760ef4f08ef0
Create Date: 2026-06-03 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1a2b3d4e5f9"
down_revision: Union[str, None] = "760ef4f08ef0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "dataset_database"
NEW_COLUMNS = (
    "cognee_version",
    "migration_revision",
)


def _existing_columns(inspector) -> set:
    return {col["name"] for col in inspector.get_columns(TABLE_NAME)}


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = _existing_columns(inspector)

    for column_name in NEW_COLUMNS:
        if column_name not in existing:
            op.add_column(TABLE_NAME, sa.Column(column_name, sa.String(), nullable=True))


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = _existing_columns(inspector)

    for column_name in NEW_COLUMNS:
        if column_name in existing:
            op.drop_column(TABLE_NAME, column_name)
