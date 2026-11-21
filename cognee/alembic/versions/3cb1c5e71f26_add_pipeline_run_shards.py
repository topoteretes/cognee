"""Add pipeline_run_shards and pipeline_run_shard_items

Revision ID: 3cb1c5e71f26
Revises: e3f1a9c5b7d2
Create Date: 2026-11-11 02:30:00.000000

A run whose data items go to workers cuts them into shards, one per worker:
one pipeline_run_shards row per shard, and one pipeline_run_shard_items row per
data item of it. The rows are how a shard reaches its worker and how the
worker's answer comes back, and they stay as the record of the run.

Idempotent: every table and index is inspector-guarded on its own, so an
upgrade or downgrade interrupted between two statements finishes on rerun.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3cb1c5e71f26"
down_revision: str | None = "e3f1a9c5b7d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SHARDS = "pipeline_run_shards"
ITEMS = "pipeline_run_shard_items"
SHARD_INDEXES = {
    "ix_pipeline_run_shards_pipeline_run_id": ["pipeline_run_id"],
    "ix_pipeline_run_shards_dataset_id": ["dataset_id"],
}
ITEM_INDEXES = {"ix_pipeline_run_shard_items_data_id": ["data_id"]}


def _inspector():
    """A fresh inspector per check: one created before a DDL statement does not see it."""
    return sa.inspect(op.get_bind())


def _create_missing_indexes(table: str, indexes: dict[str, list[str]]) -> None:
    for name, columns in indexes.items():
        if name not in {index["name"] for index in _inspector().get_indexes(table)}:
            op.create_index(name, table, columns, unique=False)


def upgrade() -> None:
    if not _inspector().has_table(SHARDS):
        op.create_table(
            SHARDS,
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("pipeline_run_id", sa.Uuid(), nullable=False),
            sa.Column("dataset_id", sa.Uuid(), nullable=False),
            sa.Column("shard_index", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("spec", sa.JSON(), nullable=False),
            sa.Column("tokens_in", sa.Integer(), nullable=True),
            sa.Column("tokens_out", sa.Integer(), nullable=True),
            sa.Column("embedding_tokens", sa.Integer(), nullable=True),
            sa.Column("error_class", sa.String(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    _create_missing_indexes(SHARDS, SHARD_INDEXES)

    if not _inspector().has_table(ITEMS):
        op.create_table(
            ITEMS,
            sa.Column("shard_id", sa.Uuid(), nullable=False),
            sa.Column("data_id", sa.Uuid(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("error_class", sa.String(), nullable=True),
            sa.Column("error_message", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["shard_id"], [f"{SHARDS}.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["data_id"], ["data.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("shard_id", "data_id"),
        )
    _create_missing_indexes(ITEMS, ITEM_INDEXES)


def downgrade() -> None:
    for table, indexes in ((ITEMS, ITEM_INDEXES), (SHARDS, SHARD_INDEXES)):
        if not _inspector().has_table(table):
            continue
        for name in indexes:
            if name in {index["name"] for index in _inspector().get_indexes(table)}:
                op.drop_index(name, table_name=table)
        op.drop_table(table)
