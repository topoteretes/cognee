"""Add expiring team invitations.

Revision ID: e7a1c9d2f4b6
Revises: a7c2e9f4b8d1
"""

import sqlalchemy as sa
from alembic import op

revision = "e7a1c9d2f4b6"
down_revision = "a7c2e9f4b8d1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "team_invitations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("created_by", sa.UUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("token_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )


def downgrade():
    op.drop_table("team_invitations")
