"""Replace graph ledger table with nodes and edges tables

Revision ID: 84e5d08260d6
Revises: 76625596c5c3
Create Date: 2025-10-30 13:36:23.226706

"""

from typing import Sequence, Union
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "84e5d08260d6"
down_revision: Union[str, None] = "76625596c5c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    table_names = inspector.get_table_names()

    if "nodes" not in table_names:
        op.create_table(
            "nodes",
            sa.Column("id", sa.UUID, primary_key=True),
            sa.Column("slug", sa.UUID, nullable=False),
            sa.Column("user_id", sa.UUID, nullable=False),
            sa.Column("data_id", sa.UUID, nullable=False),
            sa.Column("dataset_id", sa.UUID, index=True),
            sa.Column("label", sa.String()),
            sa.Column("type", sa.String(), nullable=False),
            sa.Column("attributes", sa.JSON()),
            sa.Column("indexed_fields", sa.JSON(), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
        )

    if "edges" not in table_names:
        op.create_table(
            "edges",
            sa.Column("id", sa.UUID, primary_key=True),
            sa.Column("slug", sa.UUID, nullable=False),
            sa.Column("user_id", sa.UUID, nullable=False),
            sa.Column("data_id", sa.UUID, index=True),
            sa.Column("dataset_id", sa.UUID, index=True),
            sa.Column("source_node_id", sa.UUID, nullable=False),
            sa.Column("destination_node_id", sa.UUID, nullable=False),
            sa.Column("label", sa.Text()),
            sa.Column("relationship_name", sa.Text(), nullable=False),
            sa.Column("attributes", sa.JSON()),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
        )

    existing_indexes = [index["name"] for index in inspector.get_indexes("nodes")]

    if "index_node_dataset_slug" not in existing_indexes:
        op.create_index("index_node_dataset_slug", "nodes", ["dataset_id", "slug"])

    if "index_node_dataset_data" not in existing_indexes:
        op.create_index("index_node_dataset_data", "nodes", ["dataset_id", "data_id"])


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    table_names = inspector.get_table_names()

    if "nodes" in table_names:
        op.drop_table("nodes")

    if "edges" in table_names:
        op.drop_table("edges")
