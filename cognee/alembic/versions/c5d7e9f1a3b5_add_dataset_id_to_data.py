"""add_dataset_id_to_data

Data rows become dataset-scoped: `dataset_id` names the dataset that owns the
content row, and the (dataset_id, owner_id, content_hash) index backs
add-time dedup as a LOOKUP (scoped to dataset + owner + tenant) instead of a
content-derived identity. Existing rows keep
dataset_id = NULL (legacy shared rows); they are resolved by DatasetData
membership and split copy-on-write on their first content update, so no
backfill is required here.

Revision ID: c5d7e9f1a3b5
Revises: e5a7b9c1d3f4
Create Date: 2026-08-11

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c5d7e9f1a3b5"
down_revision: Union[str, None] = "e5a7b9c1d3f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "data_dataset_content_lookup"


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def _has_index(inspector, table, name, schema=None):
    return any(index["name"] == name for index in inspector.get_indexes(table, schema=schema))


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if not _get_column(insp, "data", "dataset_id"):
        op.add_column("data", sa.Column("dataset_id", sa.UUID(), nullable=True))
        op.create_index("ix_data_dataset_id", "data", ["dataset_id"])

    insp = sa.inspect(conn)
    if not _has_index(insp, "data", INDEX_NAME):
        op.create_index(INDEX_NAME, "data", ["dataset_id", "owner_id", "content_hash"])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if _has_index(insp, "data", INDEX_NAME):
        op.drop_index(INDEX_NAME, table_name="data")
    if _has_index(insp, "data", "ix_data_dataset_id"):
        op.drop_index("ix_data_dataset_id", table_name="data")
    if _get_column(insp, "data", "dataset_id"):
        op.drop_column("data", "dataset_id")
