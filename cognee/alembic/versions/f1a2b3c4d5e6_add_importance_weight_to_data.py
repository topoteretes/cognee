"""add_importance_weight_to_data

Revision ID: f1a2b3c4d5e6
Revises: e1ec1dcb50b6
Create Date: 2026-04-07 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "b1c2d3e4f5a6"
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

    importance_weight_column = _get_column(insp, "data", "importance_weight")
    if not importance_weight_column:
        op.add_column("data", sa.Column("importance_weight", sa.Float(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    importance_weight_column = _get_column(insp, "data", "importance_weight")
    if importance_weight_column:
        op.drop_column("data", "importance_weight")
