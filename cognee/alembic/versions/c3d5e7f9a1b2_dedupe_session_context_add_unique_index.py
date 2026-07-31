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
definition, so this migration only heals pre-existing tables. It covers the
database it is connected to, and the standalone SQLite cache file (the default
sqlite cache.db lives next to the relational SQLite database) by resolving its
location the same way the runtime does. There is no runtime fallback: if a
cache database exists that this migration cannot update — including a split
non-SQLite cache via CACHE_DB_URL — the migration fails, so a deployment
cannot proceed with a cache database the upsert would break on.
"""

import logging
import os
import sqlite3
from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "c3d5e7f9a1b2"
down_revision: Union[str, None] = "b2c4d6e8f0a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "cache_session_context"
INDEX_NAME = "uq_cache_session_context_entry"
KEY_COLUMNS = ("user_id", "session_id", "entry_id")

MANUAL_STATEMENTS = (
    f"DELETE FROM {TABLE_NAME} WHERE seq NOT IN "
    f"(SELECT MAX(seq) FROM {TABLE_NAME} GROUP BY {', '.join(KEY_COLUMNS)}); "
    f"CREATE UNIQUE INDEX {INDEX_NAME} ON {TABLE_NAME} ({', '.join(KEY_COLUMNS)});"
)


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


def standalone_sqlite_cache_path(alembic_conn) -> Optional[str]:
    """Path of the standalone SQLite cache database this migration must also fix.

    Resolves the location exactly as the runtime does (CACHE_DB_URL when set,
    else a cache.db next to the relational SQLite database). Returns None when
    no separate cache database needs handling: the SQL cache backend is not in
    use, the cache lives in the database this migration is already connected
    to, or the SQLite file does not exist yet (a fresh database gets the index
    from the table definition — never create the file from a migration).

    Raises RuntimeError for a split non-SQLite cache database (CACHE_DB_URL):
    this migration cannot reach it, and proceeding would deploy a cache the
    upsert breaks on — apply the statements there manually, then re-run.
    """
    from sqlalchemy.engine import make_url

    from cognee.infrastructure.databases.cache.config import get_cache_config
    from cognee.infrastructure.databases.cache.get_cache_engine import _resolve_cache_db_url

    cache_config = get_cache_config()
    if cache_config.cache_backend not in ("sqlite", "postgres"):
        return None

    url = make_url(_resolve_cache_db_url(cache_config.cache_backend, cache_config.cache_db_url))
    alembic_url = alembic_conn.engine.url

    if url.get_backend_name() != "sqlite":
        same_database = (
            url.get_backend_name() == alembic_url.get_backend_name()
            and url.host == alembic_url.host
            and url.port == alembic_url.port
            and url.database == alembic_url.database
        )
        if same_database:
            return None
        raise RuntimeError(
            f"The session cache lives in a separate database this migration cannot "
            f"reach (CACHE_DB_URL={url.render_as_string(hide_password=True)}). "
            f"Apply the following there manually, then re-run: {MANUAL_STATEMENTS}"
        )

    if not url.database:
        return None
    path = os.path.abspath(url.database)
    if alembic_url.database and os.path.abspath(alembic_url.database) == path:
        return None
    if not os.path.exists(path):
        return None
    return path


def heal_standalone_sqlite_cache(path: str) -> None:
    """Apply the same dedupe-then-index to the standalone cache.db file.

    Any failure propagates and fails the migration: there is no runtime
    fallback, so a cache database left without the index must block the deploy.
    """
    connection = sqlite3.connect(path, timeout=30)
    try:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,)
        ).fetchone()
        index_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (INDEX_NAME,)
        ).fetchone()
        if table_exists and not index_exists:
            connection.execute(
                f"DELETE FROM {TABLE_NAME} WHERE seq NOT IN "
                f"(SELECT MAX(seq) FROM {TABLE_NAME} GROUP BY {', '.join(KEY_COLUMNS)})"
            )
            connection.execute(
                f"CREATE UNIQUE INDEX {INDEX_NAME} ON {TABLE_NAME} ({', '.join(KEY_COLUMNS)})"
            )
            connection.commit()
            logger.info("Deduped and indexed standalone SQLite cache database at %s", path)
    finally:
        connection.close()


def drop_standalone_sqlite_cache_index(path: str) -> None:
    """Downgrade mirror: the pre-index adapter appends duplicate keys, so the
    unique index must not survive a downgrade or old writes raise IntegrityError."""
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        connection.commit()
    finally:
        connection.close()


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if TABLE_NAME in inspector.get_table_names() and INDEX_NAME not in {
        index["name"] for index in inspector.get_indexes(TABLE_NAME)
    }:
        dedupe_session_context_entries(conn)
        create_session_context_unique_index(conn)

    path = standalone_sqlite_cache_path(conn)
    if path is not None:
        heal_standalone_sqlite_cache(path)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if TABLE_NAME in inspector.get_table_names() and INDEX_NAME in {
        index["name"] for index in inspector.get_indexes(TABLE_NAME)
    }:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)

    path = standalone_sqlite_cache_path(conn)
    if path is not None:
        drop_standalone_sqlite_cache_index(path)
