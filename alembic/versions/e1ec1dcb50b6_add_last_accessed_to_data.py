"""add_last_accessed_to_data

Revision ID: e1ec1dcb50b6
Revises: 211ab850ef3d
Create Date: 2025-11-04 21:45:52.642322

"""

import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e1ec1dcb50b6"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
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

    last_accessed_column = _get_column(insp, "data", "last_accessed")
    if not last_accessed_column:
        # Always create the column for schema consistency
        op.add_column("data", sa.Column("last_accessed", sa.DateTime(timezone=True), nullable=True))

        # Only initialize existing records if feature is enabled
        enable_last_accessed = os.getenv("ENABLE_LAST_ACCESSED", "false").lower() == "true"
        if enable_last_accessed:
            op.execute("UPDATE data SET last_accessed = CURRENT_TIMESTAMP")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    last_accessed_column = _get_column(insp, "data", "last_accessed")
    if last_accessed_column:
        op.drop_column("data", "last_accessed")
