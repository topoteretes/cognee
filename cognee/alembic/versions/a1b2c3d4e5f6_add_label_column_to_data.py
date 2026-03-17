"""Add label column to data table

Revision ID: a1b2c3d4e5f6
Revises: 211ab850ef3d
Create Date: 2025-11-17 17:54:32.123456

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "46a6ce2bd2b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    label_column = _get_column(insp, "data", "label")
    if not label_column:
        op.add_column("data", sa.Column("label", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("data", "label")
