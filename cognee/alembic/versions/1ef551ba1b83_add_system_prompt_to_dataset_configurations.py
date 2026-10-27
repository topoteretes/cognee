"""add system_prompt to dataset_configurations

Revision ID: 1ef551ba1b83
Revises: b7e4c2a9d6f1
Create Date: 2026-10-27 00:00:00.000000

The answer system prompt a dataset stores next to its extraction prompt; search
uses it when the caller passes no system prompt of its own.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1ef551ba1b83"
down_revision: str | None = "b7e4c2a9d6f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "dataset_configurations"
COLUMN_NAME = "system_prompt"


def _has_column(inspector) -> bool:
    return any(col["name"] == COLUMN_NAME for col in inspector.get_columns(TABLE_NAME))


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())

    if TABLE_NAME in insp.get_table_names() and not _has_column(insp):
        op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.Text(), nullable=True))


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())

    if TABLE_NAME in insp.get_table_names() and _has_column(insp):
        op.drop_column(TABLE_NAME, COLUMN_NAME)
