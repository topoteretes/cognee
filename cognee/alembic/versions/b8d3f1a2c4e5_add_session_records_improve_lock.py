"""Add the cross-worker improve lock columns to session_records

Revision ID: b8d3f1a2c4e5
Revises: a7c2e9f4b8d1
Create Date: 2026-09-09

The per-session ``improve()`` lock used to be an in-process set with no TTL,
so it excluded nothing across workers and a hung request held it forever
(SDK-593). It now lives on the session row itself:

* ``improve_lock_token`` — opaque token of the current holder, NULL when free.
* ``improve_lock_acquired_at`` — when the holder claimed it; a lock older than
  ``IMPROVE_LOCK_TTL_SECONDS`` counts as expired and may be taken over.
* ``improve_rerun_requested`` — set by a caller that found the lock busy; the
  holder runs one more watermark-driven pass before releasing.

Existing rows get NULL / NULL / false, i.e. "free, no rerun pending".
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8d3f1a2c4e5"
down_revision: str | None = "a7c2e9f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "session_records"
LOCK_TOKEN_COLUMN = "improve_lock_token"
LOCK_ACQUIRED_AT_COLUMN = "improve_lock_acquired_at"
RERUN_REQUESTED_COLUMN = "improve_rerun_requested"


def _has_column(inspector, table: str, name: str) -> bool:
    return any(col["name"] == name for col in inspector.get_columns(table))


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if not _has_column(insp, TABLE_NAME, LOCK_TOKEN_COLUMN):
        op.add_column(TABLE_NAME, sa.Column(LOCK_TOKEN_COLUMN, sa.String(), nullable=True))

    insp = sa.inspect(conn)
    if not _has_column(insp, TABLE_NAME, LOCK_ACQUIRED_AT_COLUMN):
        op.add_column(
            TABLE_NAME,
            sa.Column(LOCK_ACQUIRED_AT_COLUMN, sa.DateTime(timezone=True), nullable=True),
        )

    insp = sa.inspect(conn)
    if not _has_column(insp, TABLE_NAME, RERUN_REQUESTED_COLUMN):
        op.add_column(
            TABLE_NAME,
            sa.Column(
                RERUN_REQUESTED_COLUMN,
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    for column in (RERUN_REQUESTED_COLUMN, LOCK_ACQUIRED_AT_COLUMN, LOCK_TOKEN_COLUMN):
        insp = sa.inspect(conn)
        if _has_column(insp, TABLE_NAME, column):
            # batch mode: SQLite cannot DROP COLUMN in place.
            with op.batch_alter_table(TABLE_NAME) as batch_op:
                batch_op.drop_column(column)
