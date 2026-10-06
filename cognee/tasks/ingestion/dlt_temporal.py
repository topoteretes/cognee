"""Which cells of a DLT row are points in time, and the Timestamp each one names.

A relational source states its dates exactly — ``2024-03-01``,
``2024-03-02 10:15:00.000000`` — so no inference is needed: a cell whose whole
value is an ISO 8601 date or date-time becomes an edge from the row to the
``Timestamp`` node for that instant, named after the column. dlt types a
date-time cell as ``DATETIME``/``TIMESTAMP`` but leaves a bare date as
``TEXT`` (its ISO-date detector is off by default), so the decision is made
on the value's shape, not the column type; the shape is the same one dlt's
own detector uses. Prose, bare years and relative phrases are never read from
a cell. Primary-key and foreign-key columns are row identity and edges
already, so they are skipped like ``_selected_column_values`` skips them.

The Timestamp is built by ``timestamp_from_text`` from the normalized string,
so its id is the one every other mention of that instant resolves to: a row
dated 2024-03-01 and a document that says "1 March 2024" share one node.
"""

import re

from cognee.tasks.ingestion.dlt_row_data import DltRowData

# Date, optionally followed by a time with fractional seconds and a zone; the
# fraction and the zone are dropped, the time is kept at second precision.
_ISO_CELL = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)


def timestamp_str_for_cell(value) -> str | None:
    """``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` for an ISO-shaped cell, else None.

    The result is one of the normalized forms ``timestamp_from_text`` accepts
    without inference; a cell in the right shape that is not a real date
    (``2024-02-30``) is rejected there, not here.
    """
    if value is None:
        return None
    match = _ISO_CELL.match(str(value).strip())
    if match is None:
        return None
    date, time = match.groups()
    return f"{date} {time}" if time else date


def temporal_cells(row: DltRowData) -> dict[str, str]:
    """``{column: normalized timestamp string}`` for the row's ISO-shaped cells."""
    fk_columns = {fk.get("column", "") for fk in row.foreign_keys}
    cells = {}
    for column, value in row.row_data.items():
        if column == row.primary_key_column or column in fk_columns:
            continue
        normalized = timestamp_str_for_cell(value)
        if normalized is not None:
            cells[column] = normalized
    return cells
