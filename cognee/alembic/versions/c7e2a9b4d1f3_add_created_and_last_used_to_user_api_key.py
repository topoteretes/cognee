"""add_created_and_last_used_to_user_api_key

Revision ID: c7e2a9b4d1f3
Revises: b8c1d3e5f7a9
Create Date: 2026-08-18 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7e2a9b4d1f3"
down_revision: Union[str, None] = "b8c1d3e5f7a9"
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

    # Both columns stay nullable: existing keys have no known creation or
    # last-use time, and auth fills last_used_at lazily on first use.
    if not _get_column(insp, "user_api_key", "created_at"):
        op.add_column(
            "user_api_key", sa.Column("created_at", sa.DateTime(timezone=True), nullable=True)
        )

    if not _get_column(insp, "user_api_key", "last_used_at"):
        op.add_column(
            "user_api_key", sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if _get_column(insp, "user_api_key", "last_used_at"):
        op.drop_column("user_api_key", "last_used_at")

    if _get_column(insp, "user_api_key", "created_at"):
        op.drop_column("user_api_key", "created_at")
