"""Store the UUID columns of existing SQLite tables as text

Revision ID: c5e8a1f4b9d3
Revises: e3f1a9c5b7d2
Create Date: 2026-11-22 12:00:00.000000

SQLite gives a column declared ``UUID`` NUMERIC affinity, so a hex UUID that
SQLite reads as a number (all digits, or digits around a single ``e``) is stored
as a REAL that has lost digits or become infinity, and reading it back fails.
``sqlite_uuid`` makes new tables declare ``CHAR(32)``, but tables built before
it still declare ``UUID``. This revision rebuilds each of them with its own DDL,
``UUID`` replaced by ``CHAR(32)``, following SQLite's procedure for a column
type change: create, copy, drop, rename, then recreate the indexes and triggers
from their stored SQL. The rename leaves the table name quoted in its DDL. A
view or trigger on another table that names a rebuilt table makes the rename
fail; cognee defines none.

A UUID already stored as a number cannot be recovered. Edge evidence is an
append-only sidecar that records which chunk each graph edge came from, and
cognify writes it again when it next processes the document. Its rows holding
such a number are deleted: every read of them fails, and the stored id no
longer matches the edge, chunk or document it named. In any other table such a
value is a primary record, so the revision raises before changing anything and
names the rows for the operator to fix or delete.

Other dialects are untouched. Downgrade does nothing, since declaring ``UUID``
again would bring the corruption back.

Idempotent: a second run finds no column declared ``UUID``, and a table left by
an interrupted rebuild is dropped before it is built again.
"""

import logging
import re
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c5e8a1f4b9d3"
down_revision: str | None = "e3f1a9c5b7d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger(__name__)

EVIDENCE_TABLE = "provenance_edge_evidence"
# The type that follows a column's name, so a column named uuid keeps its name.
DECLARED_UUID = re.compile(r'([(,]\s*(?:"[^"]+"|\w+)\s+)UUID(?=[\s,)])', re.IGNORECASE)
CREATE_TABLE_PREFIX = re.compile(r'^CREATE TABLE\s+(?:"[^"]+"|\S+)\s*\(')


def _uuid_columns(connection, table: str) -> list[str]:
    return [
        row[1]
        for row in connection.exec_driver_sql(f'PRAGMA table_info("{table}")')
        if row[2].upper() == "UUID"
    ]


def _holds_number(column: str) -> str:
    return f"typeof(\"{column}\") NOT IN ('text', 'null')"


def _any_holds_number(columns: list[str]) -> str:
    return " OR ".join(_holds_number(column) for column in columns)


def _rows_holding_numbers(connection, table: str, columns: list[str]) -> str | None:
    where = _any_holds_number(columns)
    flags = ", ".join(f"max({_holds_number(column)})" for column in columns)
    count, *holds = connection.exec_driver_sql(
        f'SELECT count(*), {flags} FROM "{table}" WHERE {where}'
    ).one()
    if not count:
        return None
    named = ", ".join(column for column, held in zip(columns, holds) if held)
    return f'{table} ({named}), {count} row(s): SELECT rowid, * FROM "{table}" WHERE {where}'


def _rebuild_with_text_uuids(connection, table: str, create_sql: str) -> None:
    dependents = (
        connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master "
            "WHERE tbl_name = ? AND type IN ('index', 'trigger') AND sql IS NOT NULL",
            (table,),
        )
        .scalars()
        .all()
    )
    rebuilt = f"_uuid_as_text_{table}"
    rebuilt_sql = CREATE_TABLE_PREFIX.sub(
        f'CREATE TABLE "{rebuilt}" (', DECLARED_UUID.sub(r"\1CHAR(32)", create_sql), count=1
    )
    # The driver commits this CREATE on its own. The copy's INSERT opens a transaction that
    # the DROP and RENAME join, so a crash can leave only this scratch table behind.
    connection.exec_driver_sql(f'DROP TABLE IF EXISTS "{rebuilt}"')
    connection.exec_driver_sql(rebuilt_sql)
    connection.exec_driver_sql(f'INSERT INTO "{rebuilt}" SELECT * FROM "{table}"')
    connection.exec_driver_sql(f'DROP TABLE "{table}"')
    connection.exec_driver_sql(f'ALTER TABLE "{rebuilt}" RENAME TO "{table}"')
    for sql in dependents:
        connection.exec_driver_sql(sql)


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        return
    # Dropping a parent table with foreign keys enforced would cascade into its children.
    if connection.exec_driver_sql("PRAGMA foreign_keys").scalar():
        raise RuntimeError("Rebuilding tables needs PRAGMA foreign_keys=OFF")
    tables = [
        (table, create_sql, columns)
        for table, create_sql in connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        ).all()
        if (columns := _uuid_columns(connection, table))
    ]
    unrecoverable = [
        rows
        for table, _, columns in tables
        if table != EVIDENCE_TABLE and (rows := _rows_holding_numbers(connection, table, columns))
    ]
    if unrecoverable:
        raise RuntimeError(
            "These rows hold UUIDs that SQLite stored as numbers, which cannot be "
            "recovered. Fix or delete them, then upgrade again:\n" + "\n".join(unrecoverable)
        )
    for table, create_sql, columns in tables:
        if table == EVIDENCE_TABLE:
            deleted = connection.exec_driver_sql(
                f'DELETE FROM "{table}" WHERE {_any_holds_number(columns)}'
            ).rowcount
            if deleted:
                logger.warning(
                    "Deleted %d %s rows whose ids were stored as numbers", deleted, table
                )
        _rebuild_with_text_uuids(connection, table, create_sql)
        logger.info("Rebuilt %s to store its UUID columns as text", table)


def downgrade() -> None:
    pass
