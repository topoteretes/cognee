"""Replace graph ledger table with nodes and edges tables

Revision ID: 84e5d08260d6
Revises: 211ab850ef3d
Create Date: 2025-10-30 13:36:23.226706

"""

from uuid import NAMESPACE_OID, uuid4, uuid5
from typing import Sequence, Union
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "84e5d08260d6"
down_revision: Union[str, None] = "211ab850ef3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    table_names = inspector.get_table_names()

    if "graph_relationship_ledger" in table_names:
        op.drop_table("graph_relationship_ledger")

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
            sa.Column("indexed_fields", sa.JSON(), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
        )

    if "edges" not in table_names:
        op.create_table(
            "edges",
            sa.Column("id", sa.UUID, primary_key=True),
            sa.Column("user_id", sa.UUID, nullable=False),
            sa.Column("data_id", sa.UUID, index=True),
            sa.Column("dataset_id", sa.UUID, index=True),
            sa.Column("source_node_id", sa.UUID, nullable=False),
            sa.Column("destination_node_id", sa.UUID, nullable=False),
            sa.Column("label", sa.String()),
            sa.Column("relationship_name", sa.String(), nullable=False),
            sa.Column("props", sa.JSON()),
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

    if "graph_relationship_ledger" not in table_names:
        op.create_table(
            "graph_relationship_ledger",
            sa.Column(
                "id",
                sa.UUID,
                primary_key=True,
                default=lambda: uuid5(NAMESPACE_OID, f"{datetime.now(timezone.utc).timestamp()}"),
            ),
            sa.Column("source_node_id", sa.UUID, nullable=False),
            sa.Column("destination_node_id", sa.UUID, nullable=False),
            sa.Column("creator_function", sa.String(), nullable=False),
            sa.Column("node_label", sa.String(), nullable=False),
            sa.Column(
                "created_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
            sa.Column(
                "deleted_at", sa.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
            ),
            sa.Column("user_id", sa.UUID, nullable=False),
        )

    op.create_index("idx_graph_relationship_id", "graph_relationship_ledger", ["id"])
    op.create_index(
        "idx_graph_relationship_ledger_source_node_id",
        "graph_relationship_ledger",
        ["source_node_id"],
    )
    op.create_index(
        "idx_graph_relationship_ledger_destination_node_id",
        "graph_relationship_ledger",
        ["destination_node_id"],
    )
