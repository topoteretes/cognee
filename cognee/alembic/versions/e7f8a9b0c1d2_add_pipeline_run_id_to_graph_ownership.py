"""add pipeline_run_id to nodes and edges ownership tables

Revision ID: e7f8a9b0c1d2
Revises: d4e5f6a7b8c9
Create Date: 2026-05-11
"""

from alembic import op
import sqlalchemy as sa

revision = "e7f8a9b0c1d2"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def _has_column(insp, table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in insp.get_columns(table_name))


def _has_index(insp, table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in insp.get_indexes(table_name))


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if not _has_column(insp, "nodes", "pipeline_run_id"):
        op.add_column("nodes", sa.Column("pipeline_run_id", sa.UUID(), nullable=True))
    if not _has_index(insp, "nodes", "ix_nodes_pipeline_run_id"):
        op.create_index("ix_nodes_pipeline_run_id", "nodes", ["pipeline_run_id"], unique=False)

    if not _has_column(insp, "edges", "pipeline_run_id"):
        op.add_column("edges", sa.Column("pipeline_run_id", sa.UUID(), nullable=True))
    if not _has_index(insp, "edges", "ix_edges_pipeline_run_id"):
        op.create_index("ix_edges_pipeline_run_id", "edges", ["pipeline_run_id"], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if _has_index(insp, "edges", "ix_edges_pipeline_run_id"):
        op.drop_index("ix_edges_pipeline_run_id", table_name="edges")
    if _has_column(insp, "edges", "pipeline_run_id"):
        op.drop_column("edges", "pipeline_run_id")

    if _has_index(insp, "nodes", "ix_nodes_pipeline_run_id"):
        op.drop_index("ix_nodes_pipeline_run_id", table_name="nodes")
    if _has_column(insp, "nodes", "pipeline_run_id"):
        op.drop_column("nodes", "pipeline_run_id")
