"""incremental_loading

Revision ID: 1daae0df1866
Revises: b9274c27a25a
Create Date: 2025-08-12 13:14:12.515935

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.mutable import MutableDict

# revision identifiers, used by Alembic.
revision: str = "1daae0df1866"
down_revision: Union[str, None] = "b9274c27a25a"
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

    # If column already exists skip migration
    pipeline_status_column = _get_column(insp, "data", "pipeline_status")
    if not pipeline_status_column:
        op.add_column(
            "data",
            sa.Column(
                "pipeline_status",
                MutableDict.as_mutable(sa.JSON),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )


def downgrade() -> None:
    op.drop_column("data", "pipeline_status")
