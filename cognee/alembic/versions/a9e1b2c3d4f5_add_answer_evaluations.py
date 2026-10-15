"""Add bounded answer evaluation state.

Revision ID: a9e1b2c3d4f5
Revises: f6b8d0a2c4e6
"""

import sqlalchemy as sa
from alembic import op

revision = "a9e1b2c3d4f5"
down_revision = "f6b8d0a2c4e6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "answer_evaluation_scopes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(36), nullable=False),
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("synced_revision", sa.Integer(), nullable=False),
        sa.Column("last_run_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "ix_answer_evaluation_scopes_owner_id", "answer_evaluation_scopes", ["owner_id"]
    )
    op.create_table(
        "answer_evaluation_traces",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scope_id", sa.String(64), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("expires_at", sa.Float(), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("ciphertext", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_answer_evaluation_traces_scope_id", "answer_evaluation_traces", ["scope_id"]
    )
    op.create_table(
        "answer_evaluation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("scope_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
    )
    op.create_index("ix_answer_evaluation_runs_scope_id", "answer_evaluation_runs", ["scope_id"])


def downgrade():
    op.drop_table("answer_evaluation_runs")
    op.drop_table("answer_evaluation_traces")
    op.drop_table("answer_evaluation_scopes")
