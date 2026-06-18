"""add pipeline run id to nodes and edges

Revision ID: aa753a730673
Revises: d8f4a1b2c3e9
Create Date: 2026-06-12 16:05:38.603946

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "aa753a730673"
down_revision: Union[str, None] = "d8f4a1b2c3e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
