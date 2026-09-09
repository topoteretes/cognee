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
from sqlalchemy.engine.reflection import Inspector

revision: str = "e7f9a1c3d5b8"
down_revision: str | None = "a7c2e9f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_data_dataset_created"


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("data")]
    if INDEX_NAME not in existing_indexes:
        if conn.dialect.name == "postgresql":
            # CREATE INDEX (without CONCURRENTLY) holds a table-wide lock for
            # the whole build. `data` is one row per ingested document, so on a
            # real deployment that lock is measured in minutes -- this is the
            # migration in this repo most likely to meet a large table.
            # CONCURRENTLY cannot run inside a transaction, hence
            # autocommit_block().
            #
            # IF NOT EXISTS matters here specifically: a CONCURRENTLY build that
            # dies mid-way (deadlock/timeout/crash) leaves an INVALID index
            # under this name, and the `existing_indexes` guard above only sees
            # valid ones -- so a retry would hit "relation already exists"
            # instead of cleanly no-op'ing.
            with op.get_context().autocommit_block():
                op.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON data "
                    f"(dataset_id, created_at DESC, id)"
                )
        else:
            op.execute(f"CREATE INDEX {INDEX_NAME} ON data (dataset_id, created_at DESC, id)")


def downgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    existing_indexes = [idx["name"] for idx in inspector.get_indexes("data")]
    if INDEX_NAME in existing_indexes:
        if conn.dialect.name == "postgresql":
            with op.get_context().autocommit_block():
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        else:
            op.drop_index(INDEX_NAME, "data")
