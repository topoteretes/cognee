"""Add sync_operations table

Revision ID: a1b2c3d4e5f6
Revises: 211ab850ef3d
Create Date: 2025-11-17 17:54:32.123456

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "211ab850ef3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column(
        "data",
        sa.Column("label", sa.String(), nullable=True)),

def downgrade() -> None:
    op.drop_column("data", "label")