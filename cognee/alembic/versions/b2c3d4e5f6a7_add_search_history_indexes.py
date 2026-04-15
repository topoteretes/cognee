"""add created_at index to queries and results tables

Revision ID: b2c3d4e5f6a7
Revises:
Create Date: 2026-04-04
"""

from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_queries_created_at", "queries", ["created_at"])
    op.create_index("ix_results_created_at", "results", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_results_created_at", "results")
    op.drop_index("ix_queries_created_at", "queries")
