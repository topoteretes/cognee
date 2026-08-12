"""add_session_id_to_queries_and_results

Revision ID: a7c3e9f1b5d8
Revises: e5a7b9c1d3f4
Create Date: 2026-08-12 00:00:00.000000

Adds the caller-supplied session id a search/recall was made under, so recall
history can be filtered per agent — the session id prefix names the tool that
asked ("claude_..." from the Claude Code plugin, "codex_..." from Codex).
Nullable and forward-looking: rows written before this migration keep NULL, and
so do callers that send no session id at all.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c3e9f1b5d8"
down_revision: Union[str, None] = "e5a7b9c1d3f4"
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
        if "session_id" not in columns:
            op.add_column(table, sa.Column("session_id", sa.String(), nullable=True))

        index_name = f"ix_{table}_session_id"
        indexes = {idx["name"] for idx in insp.get_indexes(table)}
        if index_name not in indexes:
            op.create_index(index_name, table, ["session_id"])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    existing_tables = set(insp.get_table_names())

    for table in _TABLES:
        if table not in existing_tables:
            continue

        index_name = f"ix_{table}_session_id"
        indexes = {idx["name"] for idx in insp.get_indexes(table)}
        if index_name in indexes:
            op.drop_index(index_name, table_name=table)

        columns = {col["name"] for col in insp.get_columns(table)}
        if "session_id" in columns:
            op.drop_column(table, "session_id")
