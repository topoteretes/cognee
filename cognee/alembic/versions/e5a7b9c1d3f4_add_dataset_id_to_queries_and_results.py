"""add_dataset_id_to_queries_and_results

Revision ID: e5a7b9c1d3f4
Revises: d4e6f8a0b2c3
Create Date: 2026-08-10 00:00:00.000000

Adds the dataset a search/recall was scoped to, so recall history can be
filtered per dataset. Nullable and forward-looking: rows written before this
migration keep NULL, and so do searches that spanned more than one dataset.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5a7b9c1d3f4"
down_revision: Union[str, None] = "d4e6f8a0b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = ("queries", "results")


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    existing_tables = set(insp.get_table_names())

    for table in _TABLES:
        if table not in existing_tables:
            continue

        columns = {col["name"] for col in insp.get_columns(table)}
        if "dataset_id" not in columns:
            op.add_column(table, sa.Column("dataset_id", sa.UUID(), nullable=True))

        index_name = f"ix_{table}_dataset_id"
        indexes = {idx["name"] for idx in insp.get_indexes(table)}
        if index_name not in indexes:
            op.create_index(index_name, table, ["dataset_id"])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    existing_tables = set(insp.get_table_names())

    for table in _TABLES:
        if table not in existing_tables:
            continue

        index_name = f"ix_{table}_dataset_id"
        indexes = {idx["name"] for idx in insp.get_indexes(table)}
        if index_name in indexes:
            op.drop_index(index_name, table_name=table)

        columns = {col["name"] for col in insp.get_columns(table)}
        if "dataset_id" in columns:
            op.drop_column(table, "dataset_id")
