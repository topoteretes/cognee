"""Drop the data key from pipeline_runs.run_info (COG-6089)

Revision ID: b7e4c2a9d6f1
Revises: d4a8e2f6b1c9
Create Date: 2026-10-22 00:00:00.000000

Until COG-6089 the pipeline-run writers stored the run's input in
``run_info["data"]``: for text passed to ``add()`` the first 512 characters of
the text itself, so user content sat in the audit table and no ``forget()``
mode removed it; for cognify the id of every document in the dataset, on the
STARTED row and again on the terminal row, so the table grew with dataset size
times run rate. Nothing ever read the key. The writers no longer store it;
this revision removes it from the rows already stored.

Every ``run_info`` holding a ``data`` key (and ``errored_data``, written by a
pre-release build) loses those keys. Every other key — ``recovery_lock``,
``progress``, ``error``, the improve stamps — is kept, and no row is deleted.
Only rows holding the key are read: SQLite and Postgres filter in SQL, any
other dialect scans the table in Python. Idempotent, since a second run finds
no such row. Irreversible by design: the removed values are what the revision
exists to remove, so ``downgrade`` is a no-op. Space is returned to the
operating system only by the database's own VACUUM.
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e4c2a9d6f1"
down_revision: str | None = "d4a8e2f6b1c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "pipeline_runs"
DROPPED_KEYS = ("data", "errored_data")
BATCH_SIZE = 500


def upgrade() -> None:
    bind = op.get_bind()
    if TABLE_NAME not in sa.inspect(bind).get_table_names():
        return
    table = _table()
    rows = rows_with_dropped_keys(bind, table)
    statement = (
        sa.update(table)
        .where(table.c.id == sa.bindparam("row_id"))
        .values(run_info=sa.bindparam("new_run_info"))
    )
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        bind.execute(
            statement,
            [{"row_id": row_id, "new_run_info": run_info} for row_id, run_info in batch],
        )


def downgrade() -> None:
    """Nothing to restore: the removed values are gone on purpose."""


def _table() -> sa.TableClause:
    return sa.table(TABLE_NAME, sa.column("id", sa.Uuid()), sa.column("run_info", sa.JSON()))


def dropped_key_filter(dialect_name: str) -> sa.TextClause | None:
    """SQL predicate for rows whose ``run_info`` holds a key to drop, per dialect.

    ``None`` means the dialect has no known JSON key test, so every row is read
    and filtered in Python.
    """
    if dialect_name == "sqlite":
        return sa.text(
            "json_type(run_info, '$.data') IS NOT NULL "
            "OR json_type(run_info, '$.errored_data') IS NOT NULL"
        )
    if dialect_name == "postgresql":
        return sa.text("run_info::jsonb ?| array['data', 'errored_data']")
    return None


def rows_with_dropped_keys(bind, table: sa.TableClause) -> list[tuple[Any, dict]]:
    """``(id, run_info without the dropped keys)`` for every row holding one."""
    query = sa.select(table.c.id, table.c.run_info)
    predicate = dropped_key_filter(bind.dialect.name)
    if predicate is not None:
        query = query.where(predicate)
    stripped = []
    for row_id, run_info in bind.execute(query):
        # The Python check also guards the dialect filter: a row the SQL predicate
        # let through that holds no such key is left untouched.
        if isinstance(run_info, dict) and any(key in run_info for key in DROPPED_KEYS):
            stripped.append(
                (row_id, {key: value for key, value in run_info.items() if key not in DROPPED_KEYS})
            )
    return stripped
