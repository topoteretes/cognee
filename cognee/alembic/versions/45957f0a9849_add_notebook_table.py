"""Add notebook table

Revision ID: 45957f0a9849
Revises: 9e7a3cb85175
Create Date: 2025-09-10 17:47:58.201319

"""

from datetime import datetime, timezone
from uuid import uuid4
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "45957f0a9849"
down_revision: Union[str, None] = "9e7a3cb85175"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "notebooks" not in inspector.get_table_names():
        # Define table with all necessary columns including primary key
        op.create_table(
            "notebooks",
            sa.Column("id", sa.UUID, primary_key=True, default=uuid4),  # Critical for SQLite
            sa.Column("owner_id", sa.UUID, index=True),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("cells", sa.JSON(), nullable=False),
            sa.Column("deletable", sa.Boolean(), default=True),
            sa.Column("created_at", sa.DateTime(), default=lambda: datetime.now(timezone.utc)),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "notebooks" in inspector.get_table_names():
        op.drop_table("notebooks")
