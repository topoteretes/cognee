"""Add label column to data table for custom naming

Revision ID: f5a1b2c3d4e5
Revises: e4ebee1091e7
Create Date: 2025-12-11 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f5a1b2c3d4e5"
down_revision: Union[str, None] = "e4ebee1091e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    # Check if label column already exists
    data_columns = {col["name"] for col in insp.get_columns("data")}

    if "label" not in data_columns:
        op.add_column("data", sa.Column("label", sa.String(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    # Check if label column exists before dropping
    data_columns = {col["name"] for col in insp.get_columns("data")}

    if "label" in data_columns:
        op.drop_column("data", "label")
