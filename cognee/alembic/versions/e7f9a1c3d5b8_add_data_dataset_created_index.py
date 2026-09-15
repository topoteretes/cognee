"""Add composite index on data for the dataset listing

Revision ID: e7f9a1c3d5b8
Revises: a7c2e9f4b8d1
Create Date: 2026-09-09

get_dataset_data.py filters on dataset_id and orders newest-first. dataset_id
has its own single-column index, but the filter+sort as a whole was not covered
by one, so every listing seq-scanned `data` and sorted it -- externally,
spilling to disk, once the dataset was large. A LIMIT does not avoid that: the
whole partition still has to be sorted to find the top N. Measured on a
171,828-document dataset, the listing endpoint took 408 s.

The ordering was previously data_size DESC, on a column with no index and which
DataDTO does not even expose. It is now created_at DESC, id -- newest first,
tiebroken so that paging is stable when timestamps collide -- which this index
serves end to end.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import inspect, text

revision: str = "e7f9a1c3d5b8"
down_revision: str | None = "a7c2e9f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_data_dataset_created"


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        # Reflection and IF NOT EXISTS only establish existence, not validity.
        # A failed concurrent build leaves an index that the planner cannot use.
        valid = conn.execute(
            text(
                "SELECT i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE i.indrelid = 'data'::regclass AND c.relname = :name"
            ),
            {"name": INDEX_NAME},
        ).scalar_one_or_none()
        if valid is True:
            return
        # Concurrent DDL cannot run inside a transaction. Drop an interrupted
        # build before retrying, so success always leaves a usable index.
        with op.get_context().autocommit_block():
            if valid is False:
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
            op.execute(
                f"CREATE INDEX CONCURRENTLY {INDEX_NAME} ON data (dataset_id, created_at DESC NULLS LAST, id)"
            )
    elif INDEX_NAME not in {idx["name"] for idx in inspect(conn).get_indexes("data")}:
        op.execute(f"CREATE INDEX {INDEX_NAME} ON data (dataset_id, created_at DESC, id)")


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("data")]
    if INDEX_NAME in existing_indexes:
        if conn.dialect.name == "postgresql":
            with op.get_context().autocommit_block():
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        else:
            op.drop_index(INDEX_NAME, "data")
