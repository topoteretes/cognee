"""Schema definitions for the SQL cache adapter (engine-agnostic: Postgres + SQLite).

Tables live on a private MetaData (``cache_metadata``) — not the relational
declarative Base and not alembic-managed — mirroring the Postgres graph adapter's
create-on-init pattern. Payload columns use JSONB on Postgres and plain JSON on
SQLite; seq primary keys degrade from BIGINT identity to INTEGER autoincrement
on SQLite so unit tests can run on aiosqlite without a server.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    JSON,
    MetaData,
    Table,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

cache_metadata = MetaData()


class StringKey(TypeDecorator):
    """TEXT key column that coerces stringable ids (e.g. ``uuid.UUID``) to ``str``.

    Client-side only: DDL is delegated to ``impl`` so the emitted column is a
    plain TEXT, identical to before — existing tables need no migration. The
    asyncpg dialect renders explicit bind casts, so an id bound as a
    ``uuid.UUID`` would make Postgres parse ``text = uuid`` and raise 42883
    ("operator does not exist") instead of coercing; sqlite rejects non-str
    binds outright. Normalizing in the bind processor keeps every read and
    write keyed by the same string no matter which type the caller holds.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None or isinstance(value, str):
            return value
        return str(value)

    def coerce_compared_value(self, op, value):
        # Compare through this decorator (not the raw impl) so binds in
        # WHERE/IN comparisons run through process_bind_param too.
        return self


def _payload_type():
    """JSONB on Postgres, generic JSON on SQLite."""
    return JSONB().with_variant(JSON(), "sqlite")


def _seq_type():
    """BIGINT identity on Postgres, INTEGER autoincrement on SQLite."""
    return BigInteger().with_variant(Integer(), "sqlite")


# QA entries: one row per entry; qa_id promoted to a column for direct UPDATE.
cache_qa_entries = Table(
    "cache_qa_entries",
    cache_metadata,
    Column("seq", _seq_type(), primary_key=True, autoincrement=True),
    Column("user_id", StringKey(), nullable=False),
    Column("session_id", StringKey(), nullable=False),
    Column("qa_id", StringKey(), nullable=False),
    Column("payload", _payload_type(), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("user_id", "session_id", "qa_id", name="uq_cache_qa_entry"),
)

Index(
    "idx_cache_qa_session",
    cache_qa_entries.c.user_id,
    cache_qa_entries.c.session_id,
    cache_qa_entries.c.seq,
)
Index(
    "idx_cache_qa_expires",
    cache_qa_entries.c.expires_at,
    postgresql_where=cache_qa_entries.c.expires_at.isnot(None),
    sqlite_where=cache_qa_entries.c.expires_at.isnot(None),
)

# Agent traces: append-only.
cache_trace_entries = Table(
    "cache_trace_entries",
    cache_metadata,
    Column("seq", _seq_type(), primary_key=True, autoincrement=True),
    Column("user_id", StringKey(), nullable=False),
    Column("session_id", StringKey(), nullable=False),
    Column("payload", _payload_type(), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)

Index(
    "idx_cache_trace_session",
    cache_trace_entries.c.user_id,
    cache_trace_entries.c.session_id,
    cache_trace_entries.c.seq,
)
Index(
    "idx_cache_trace_expires",
    cache_trace_entries.c.expires_at,
    postgresql_where=cache_trace_entries.c.expires_at.isnot(None),
    sqlite_where=cache_trace_entries.c.expires_at.isnot(None),
)

# Session-context entries: append-only, kind-discriminated ("context"/"feedback").
# entry_id is promoted from the payload's "id" to a column for direct UPDATE,
# mirroring how cache_qa_entries promotes qa_id.
cache_session_context = Table(
    "cache_session_context",
    cache_metadata,
    Column("seq", _seq_type(), primary_key=True, autoincrement=True),
    Column("user_id", StringKey(), nullable=False),
    Column("session_id", StringKey(), nullable=False),
    Column("entry_id", StringKey(), nullable=False),
    Column("payload", _payload_type(), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)

Index(
    "idx_cache_session_context_session",
    cache_session_context.c.user_id,
    cache_session_context.c.session_id,
    cache_session_context.c.seq,
)
Index(
    "idx_cache_session_context_expires",
    cache_session_context.c.expires_at,
    postgresql_where=cache_session_context.c.expires_at.isnot(None),
    sqlite_where=cache_session_context.c.expires_at.isnot(None),
)

# Usage logs (Redis key {log_key}:{user_id} analogue).
cache_usage_logs = Table(
    "cache_usage_logs",
    cache_metadata,
    Column("seq", _seq_type(), primary_key=True, autoincrement=True),
    Column("log_key", StringKey(), nullable=False),
    Column("user_id", StringKey(), nullable=False),
    Column("payload", _payload_type(), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)

Index(
    "idx_cache_usage",
    cache_usage_logs.c.log_key,
    cache_usage_logs.c.user_id,
    cache_usage_logs.c.seq,
)

# Generic string KV storage for small cache values.
cache_kv = Table(
    "cache_kv",
    cache_metadata,
    Column("key", StringKey(), primary_key=True),
    Column("value", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)
