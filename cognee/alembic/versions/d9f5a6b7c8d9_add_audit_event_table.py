"""add_audit_event_table

Revision ID: d9f5a6b7c8d9
Revises: d8f4a1b2c3e9
Create Date: 2026-07-04 12:20:00.000000

"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy import UUID
from datetime import datetime, timezone
from uuid import uuid4

# revision identifiers, used by Alembic.
revision: str = "d9f5a6b7c8d9"
down_revision: Union[str, None] = "d8f4a1b2c3e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "governance_audit_event",
        sa.Column("id", UUID, primary_key=True, default=uuid4),
        sa.Column("actor_id", UUID, nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("target_dataset_id", UUID, nullable=True),
        sa.Column("outcome", sa.String(10), nullable=False),
        sa.Column("policy_id", UUID, nullable=True),
        sa.Column("denial_reason", sa.Text, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("previous_hash", sa.String(64), nullable=True),
        sa.Column("row_hash", sa.String(64), nullable=True),
        sa.CheckConstraint("outcome IN ('ALLOWED', 'DENIED')", name="chk_outcome")
    )
    
    op.create_index("ix_gae_dataset_ts", "governance_audit_event", ["target_dataset_id", sa.text("timestamp DESC")])
    op.create_index("ix_gae_actor_ts", "governance_audit_event", ["actor_id", sa.text("timestamp DESC")])
    op.create_index("ix_gae_outcome", "governance_audit_event", ["outcome", sa.text("timestamp DESC")])


def downgrade() -> None:
    op.drop_index("ix_gae_outcome", table_name="governance_audit_event")
    op.drop_index("ix_gae_actor_ts", table_name="governance_audit_event")
    op.drop_index("ix_gae_dataset_ts", table_name="governance_audit_event")
    op.drop_table("governance_audit_event")
