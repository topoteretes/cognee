"""Move pipeline_runs.session_id into pipeline_run_sessions (CLO-1105)

Revision ID: dd760c09e075
Revises: b7e4c2a9d6f1
Create Date: 2026-10-21 15:54:34.739646

A run launched by an operation bound to several sessions (an improve over two
sessions) belongs to each of them, which one pipeline_runs.session_id column
cannot hold. pipeline_run_sessions holds one row per (pipeline_runs row,
session) and becomes the only place a row's sessions are stored: every
existing session_id is copied into it, then the column and its index are
dropped. "The runs of session X" is then one indexed equality lookup on every
relational backend, with no JSON membership query.

Idempotent: each step is inspector-guarded, like the other revisions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd760c09e075"
down_revision: str | None = "b7e4c2a9d6f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "pipeline_run_sessions"
INDEX_NAME = "ix_pipeline_run_sessions_session_id"
RUNS_TABLE = "pipeline_runs"
OLD_COLUMN = "session_id"
OLD_INDEX = f"ix_{RUNS_TABLE}_{OLD_COLUMN}"


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    if not insp.has_table(TABLE_NAME):
        op.create_table(
            TABLE_NAME,
            sa.Column("pipeline_run_row_id", sa.Uuid(), nullable=False),
            sa.Column(OLD_COLUMN, sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("pipeline_run_row_id", OLD_COLUMN),
        )
        op.create_index(INDEX_NAME, TABLE_NAME, [OLD_COLUMN], unique=False)

    if insp.has_table(RUNS_TABLE) and _has_column(insp, RUNS_TABLE, OLD_COLUMN):
        op.execute(
            f"INSERT INTO {TABLE_NAME} (pipeline_run_row_id, {OLD_COLUMN}) "
            f"SELECT id, {OLD_COLUMN} FROM {RUNS_TABLE} WHERE {OLD_COLUMN} IS NOT NULL"
        )
        if _has_index(insp, RUNS_TABLE, OLD_INDEX):
            op.drop_index(OLD_INDEX, table_name=RUNS_TABLE)
        _drop_column(RUNS_TABLE, OLD_COLUMN)


def downgrade() -> None:
    """Restore the column with the session of every row that has exactly one; a
    row with several sessions keeps none (the column cannot hold them)."""
    insp = sa.inspect(op.get_bind())
    if insp.has_table(RUNS_TABLE) and not _has_column(insp, RUNS_TABLE, OLD_COLUMN):
        _add_column(RUNS_TABLE, sa.Column(OLD_COLUMN, sa.String(), nullable=True))
        op.create_index(OLD_INDEX, RUNS_TABLE, [OLD_COLUMN], unique=False)
        if insp.has_table(TABLE_NAME):
            op.execute(
                f"UPDATE {RUNS_TABLE} SET {OLD_COLUMN} = ("
                f"SELECT MIN(s.{OLD_COLUMN}) FROM {TABLE_NAME} s "
                f"WHERE s.pipeline_run_row_id = {RUNS_TABLE}.id "
                f"GROUP BY s.pipeline_run_row_id HAVING COUNT(*) = 1)"
            )
    if insp.has_table(TABLE_NAME):
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
        op.drop_table(TABLE_NAME)


def _has_column(inspector, table: str, name: str) -> bool:
    return any(column["name"] == name for column in inspector.get_columns(table))


def _has_index(inspector, table: str, name: str) -> bool:
    return any(index["name"] == name for index in inspector.get_indexes(table))


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
