"""add conversation_id to queries table

Revision ID: e5f6a7b8c9d0
Revises: b2c3d4e5f6a7, d4e5f6a7b8c9
Create Date: 2026-05-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

revision = "e5f6a7b8c9d0"
down_revision = ("b2c3d4e5f6a7", "d4e5f6a7b8c9")
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_columns = [col["name"] for col in inspector.get_columns("queries")]
    if "conversation_id" not in existing_columns:
        op.add_column("queries", sa.Column("conversation_id", sa.UUID(), nullable=True))

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_conversation_id" not in existing_indexes:
        op.create_index("ix_queries_conversation_id", "queries", ["conversation_id"])


def downgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("queries")]
    if "ix_queries_conversation_id" in existing_indexes:
        op.drop_index("ix_queries_conversation_id", table_name="queries")

    existing_columns = [col["name"] for col in inspector.get_columns("queries")]
    if "conversation_id" in existing_columns:
        op.drop_column("queries", "conversation_id")
