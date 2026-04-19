"""add_session_records_table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-04-19 21:40:00.000000

Narrow session-lifecycle table. The session cache (Redis / FS) still
owns QA content and trace steps; this table only carries lifecycle +
aggregate counters so the dashboard can ORDER BY / SUM efficiently.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "session_records" in insp.get_table_names():
        return

    op.create_table(
        "session_records",
        sa.Column("session_id", sa.String, primary_key=True),
        sa.Column("user_id", sa.UUID, primary_key=True, index=True),
        sa.Column("dataset_id", sa.UUID, nullable=True, index=True),
        sa.Column(
            "status",
            sa.String,
            nullable=False,
            server_default="running",
            index=True,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("error_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_model", sa.Text, nullable=True),
    )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "session_records" in insp.get_table_names():
        op.drop_table("session_records")
