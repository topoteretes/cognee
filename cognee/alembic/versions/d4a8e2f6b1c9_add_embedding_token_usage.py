"""Add embedding token usage columns (SDK-1077)

Revision ID: d4a8e2f6b1c9
Revises: f6b8d0a2c4e6
Create Date: 2026-10-13 00:00:00.000000

Embedding tokens are stored apart from LLM tokens, never added into them:
pipeline_runs.embedding_tokens per operation (nullable, NULL = not measured,
like tokens_in), session_records.embedding_tokens / embedding_cost_usd per
session, and session_model_usage.kind, so an embedding model's per-model row
is told apart from an LLM's. Session columns are NOT NULL with a server
default that is also correct for existing rows: embeddings were never counted
before, and every existing per-model row is an LLM's.

Idempotent: every add/drop is inspector-guarded, like the other revisions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4a8e2f6b1c9"
down_revision: str | None = "f6b8d0a2c4e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column name, SQLAlchemy type factory, nullable, server default)
_NEW_COLUMNS = [
    ("pipeline_runs", "embedding_tokens", lambda: sa.Integer(), True, None),
    ("session_records", "embedding_tokens", lambda: sa.Integer(), False, "0"),
    ("session_records", "embedding_cost_usd", lambda: sa.Float(), False, "0"),
    ("session_model_usage", "kind", lambda: sa.String(), False, "llm"),
]


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    tables = set(insp.get_table_names())

    for table, name, make_type, nullable, server_default in _NEW_COLUMNS:
        if table in tables and not _has_column(insp, table, name):
            column = sa.Column(name, make_type(), nullable=nullable, server_default=server_default)
            _add_column(table, column)


def downgrade() -> None:
    insp = sa.inspect(op.get_bind())
    tables = set(insp.get_table_names())

    for table, name, *_ in _NEW_COLUMNS:
        if table in tables and _has_column(insp, table, name):
            _drop_column(table, name)


def _has_column(inspector, table: str, name: str) -> bool:
    return any(column["name"] == name for column in inspector.get_columns(table))


def _add_column(table: str, column: sa.Column) -> None:
    if op.get_context().dialect.name == "sqlite":
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(column)
    else:
        op.add_column(table, column)


def _drop_column(table: str, name: str) -> None:
    if op.get_context().dialect.name == "sqlite":
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_column(name)
    else:
        op.drop_column(table, name)
