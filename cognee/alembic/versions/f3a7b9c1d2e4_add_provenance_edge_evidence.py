"""Add the compact graph edge evidence sidecar.

Revision ID: f3a7b9c1d2e4
Revises: aa753a730673
Create Date: 2026-07-17 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f3a7b9c1d2e4"
down_revision: Union[str, None] = "aa753a730673"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "provenance_edge_evidence"
INDEXES = (
    ("ix_prov_evidence_edge", ["dataset_id", "edge_id"]),
    ("ix_prov_evidence_source", ["dataset_id", "data_id", "chunk_id"]),
    ("ix_prov_evidence_run", ["pipeline_run_id"]),
)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(TABLE_NAME):
        return

    op.create_table(
        TABLE_NAME,
        sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
        sa.Column("tenant_id", sa.UUID(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("data_id", sa.UUID(), nullable=False),
        sa.Column("pipeline_run_id", sa.UUID(), nullable=True),
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column("edge_id", sa.UUID(), nullable=False),
        sa.Column("source_node_id", sa.UUID(), nullable=False),
        sa.Column("destination_node_id", sa.UUID(), nullable=False),
        sa.Column("relationship_name", sa.Text(), nullable=False),
        sa.Column(
            "evidence_kind",
            sa.String(length=32),
            nullable=False,
            server_default="extracted",
        ),
        sa.Column("source_task", sa.String(length=255), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for index_name, columns in INDEXES:
        op.create_index(index_name, TABLE_NAME, columns, unique=False)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table(TABLE_NAME):
        op.drop_table(TABLE_NAME)
