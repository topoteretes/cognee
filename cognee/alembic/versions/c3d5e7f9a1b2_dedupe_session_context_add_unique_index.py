"""dedupe cache_session_context and add unique entry index

Revision ID: c3d5e7f9a1b2
Revises: b2c4d6e8f0a1
Create Date: 2026-07-27 00:00:00.000000

One-time cleanup for the SQL session-cache backend: duplicate
(user_id, session_id, entry_id) rows accumulated because
create_session_context_entry() appended instead of upserting (racing
update-then-create writers, e.g. the session persist watermark). Keep the
newest row per key — updates rewrite payload in place, so max(seq) carries the
latest state — then add the unique index the adapter's ON CONFLICT upsert
targets.

Cache tables are create-on-init (cognee/infrastructure/databases/cache/sql/
tables.py), not alembic-managed: fresh databases get the index from the table
definition, and SqlCacheAdapter._heal_session_context_unique_index applies the
same dedupe-then-index on init to any cache database alembic cannot reach (the
default sqlite cache.db is a separate file; CACHE_DB_URL can point anywhere).
This migration is belt-and-suspenders for cache tables living in the
alembic-managed relational database, healing them even before the adapter's
first write.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d5e7f9a1b2"
down_revision: Union[str, None] = "b2c4d6e8f0a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "cache_session_context"
INDEX_NAME = "uq_cache_session_context_entry"
KEY_COLUMNS = ("user_id", "session_id", "entry_id")


def dedupe_session_context_entries(conn) -> None:
    """Drop all but the newest (max seq) row per (user_id, session_id, entry_id)."""
    table = sa.table(
        TABLE_NAME,
        sa.column("seq"),
        sa.column("user_id"),
        sa.column("session_id"),
        sa.column("entry_id"),
    )
    newest = sa.select(sa.func.max(table.c.seq)).group_by(
        table.c.user_id, table.c.session_id, table.c.entry_id
    )
    conn.execute(sa.delete(table).where(table.c.seq.notin_(newest)))


def create_session_context_unique_index(conn) -> None:
    """Create the unique index the adapter's ON CONFLICT upsert targets."""
    conn.execute(
        sa.text(f"CREATE UNIQUE INDEX {INDEX_NAME} ON {TABLE_NAME} ({', '.join(KEY_COLUMNS)})")
    )


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if TABLE_NAME not in inspector.get_table_names():
        return
    if INDEX_NAME in {index["name"] for index in inspector.get_indexes(TABLE_NAME)}:
        return

    dedupe_session_context_entries(conn)
    create_session_context_unique_index(conn)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if TABLE_NAME not in inspector.get_table_names():
        return
    if INDEX_NAME in {index["name"] for index in inspector.get_indexes(TABLE_NAME)}:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
