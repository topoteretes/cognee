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
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

cache_metadata = MetaData()


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
    Column("user_id", Text, nullable=False),
    Column("session_id", Text, nullable=False),
    Column("qa_id", Text, nullable=False),
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
    Column("user_id", Text, nullable=False),
    Column("session_id", Text, nullable=False),
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

# Usage logs (Redis key {log_key}:{user_id} analogue).
cache_usage_logs = Table(
    "cache_usage_logs",
    cache_metadata,
    Column("seq", _seq_type(), primary_key=True, autoincrement=True),
    Column("log_key", Text, nullable=False),
    Column("user_id", Text, nullable=False),
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

# String KV: graph_knowledge:{u}:{s}, graph_sync_checkpoint:{u}:{d}:{s} — keys kept verbatim.
cache_kv = Table(
    "cache_kv",
    cache_metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)
