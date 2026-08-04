"""Add telemetry_events table

Backs the local telemetry sink (``TELEMETRY_SINK=postgres``). Unused by
deployments on the default HTTP sink, so creating it is harmless there.

Revision ID: a3f7c1b9d2e4
Revises: c3d5e7f9a1b2
Create Date: 2026-08-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "a3f7c1b9d2e4"
down_revision: Union[str, None] = "c3d5e7f9a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    insp = inspect(op.get_bind())
    if "telemetry_events" in insp.get_table_names():
        return

    op.create_table(
        "telemetry_events",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("event_name", sa.String(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("tenant_id", sa.UUID(), nullable=True),
        sa.Column("anonymous_id", sa.String(), nullable=True),
        sa.Column("properties", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_telemetry_events_event_name", "telemetry_events", ["event_name"])
    op.create_index("ix_telemetry_events_user_id", "telemetry_events", ["user_id"])
    op.create_index("ix_telemetry_events_tenant_id", "telemetry_events", ["tenant_id"])
    # The retention prune and every read path filter on created_at.
    op.create_index("ix_telemetry_events_created_at", "telemetry_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("telemetry_events")
