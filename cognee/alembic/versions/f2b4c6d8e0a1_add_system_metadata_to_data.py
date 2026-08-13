"""add_system_metadata_to_data

Revision ID: f2b4c6d8e0a1
Revises: d6e8f0a2b4c6
Create Date: 2026-08-12 00:00:00.000000

Splits system-derived metadata out of the user-writable external_metadata
field. Pipeline routing, purge scoping, and DLT orphan cleanup key on
Data.system_metadata from this revision on — external_metadata is the user's
free-form field and is never consulted for system decisions.

Existing rows whose external_metadata carries a DLT stamp
(source == "dlt" or "dlt_source") were written entirely by cognee, so the
whole dict moves to system_metadata and external_metadata is cleared.
"""

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f2b4c6d8e0a1"
down_revision: Union[str, None] = "d6e8f0a2b4c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_DLT_SOURCES = ("dlt", "dlt_source")

_data_table = sa.table(
    "data",
    sa.column("id"),
    sa.column("external_metadata", sa.JSON()),
    sa.column("system_metadata", sa.JSON()),
)


def _parse(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "data" not in set(insp.get_table_names()):
        return

    columns = {col["name"] for col in insp.get_columns("data")}
    if "system_metadata" not in columns:
        op.add_column("data", sa.Column("system_metadata", sa.JSON(), nullable=True))

    # Move DLT stamps written by earlier versions out of the user field.
    rows = conn.execute(
        sa.select(_data_table.c.id, _data_table.c.external_metadata).where(
            _data_table.c.external_metadata.isnot(None)
        )
    ).fetchall()

    for row_id, external_metadata in rows:
        parsed = _parse(external_metadata)
        if parsed is None or parsed.get("source") not in _DLT_SOURCES:
            continue
        conn.execute(
            _data_table.update()
            .where(_data_table.c.id == row_id)
            .values(system_metadata=parsed, external_metadata=None)
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "data" not in set(insp.get_table_names()):
        return

    columns = {col["name"] for col in insp.get_columns("data")}
    if "system_metadata" not in columns:
        return

    # Move DLT stamps back into external_metadata, then drop the column.
    rows = conn.execute(
        sa.select(_data_table.c.id, _data_table.c.system_metadata).where(
            _data_table.c.system_metadata.isnot(None)
        )
    ).fetchall()

    for row_id, system_metadata in rows:
        parsed = _parse(system_metadata)
        if parsed is None:
            continue
        conn.execute(
            _data_table.update().where(_data_table.c.id == row_id).values(external_metadata=parsed)
        )

    op.drop_column("data", "system_metadata")
