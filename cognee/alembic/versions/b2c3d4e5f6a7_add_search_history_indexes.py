"""add created_at index to queries and results tables

Revision ID: b2c3d4e5f6a7
Revises:
Create Date: 2026-04-04
"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector

revision = "b2c3d4e5f6a7"
down_revision = "6ee94db7381f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    # Get existing indexes for 'queries'
    queries_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_created_at" not in queries_indexes:
        op.create_index("ix_queries_created_at", "queries", ["created_at"])

    # Get existing indexes for 'results'
    results_indexes = [idx["name"] for idx in inspector.get_indexes("results")]
    if "ix_results_created_at" not in results_indexes:
        op.create_index("ix_results_created_at", "results", ["created_at"])


def downgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    # Get existing indexes for 'results'
    results_indexes = [idx["name"] for idx in inspector.get_indexes("results")]
    if "ix_results_created_at" in results_indexes:
        op.drop_index("ix_results_created_at", "results")

    # Get existing indexes for 'queries'
    queries_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_created_at" in queries_indexes:
        op.drop_index("ix_queries_created_at", "queries")
