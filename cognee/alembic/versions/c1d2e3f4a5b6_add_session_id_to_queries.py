"""add session_id to queries table

Revision ID: c1d2e3f4a5b6
Revises: 760ef4f08ef0
Create Date: 2026-06-01

"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector
import sqlalchemy as sa

revision = "c1d2e3f4a5b6"
down_revision = "760ef4f08ef0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_columns = [col["name"] for col in inspector.get_columns("queries")]
    if "session_id" not in existing_columns:
        op.add_column("queries", sa.Column("session_id", sa.String(), nullable=True))

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_session_id" not in existing_indexes:
        op.create_index("ix_queries_session_id", "queries", ["session_id"])


def downgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_session_id" in existing_indexes:
        op.drop_index("ix_queries_session_id", "queries")

    existing_columns = [col["name"] for col in inspector.get_columns("queries")]
    if "session_id" in existing_columns:
        op.drop_column("queries", "session_id")
