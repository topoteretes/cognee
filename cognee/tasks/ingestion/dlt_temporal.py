"""Which cells of a DLT row are points in time, and the Timestamp each one names.

A relational source states its dates exactly — ``2024-03-01``,
``2024-03-02 10:15:00.000000``, ``2024-03`` — so no inference is needed: a
cell whose whole value is an ISO 8601 month, date or date-time becomes an
edge from the row to the ``Timestamp`` node for that period, named after the
column. dlt types a date-time cell as ``DATETIME``/``TIMESTAMP`` but leaves a
bare date or month as ``TEXT`` (its ISO-date detector is off by default), so
the decision is made on the value's shape, not the column type. Prose,
relative phrases and bare years are never read from a cell (any four-digit
number would match a year). Foreign-key columns are skipped — they are edges
to other rows already — but the primary key is not: a time series is keyed
by its date (``dlt`` falls back to the first column), and that date is what
the row is about.

The Timestamp is built by ``timestamp_from_text`` from the normalized string,
so its id is the one every other mention of that instant resolves to: a row
dated 2024-03-01 and a document that says "1 March 2024" share one node.
"""

import re

from dlt.common.time import ensure_pendulum_datetime_utc

from cognee.tasks.ingestion.dlt_row_data import DltRowData

# A month, or a date optionally followed by a time with fractional seconds and
# a zone. The fraction is dropped (second precision); a zone is applied, not
# dropped — see ``timestamp_str_for_cell``.
_ISO_CELL = re.compile(
    r"^(\d{4}-\d{2}(?:-\d{2})?)(?:[ T](\d{2}:\d{2}:\d{2})(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)


def timestamp_str_for_cell(value) -> str | None:
    """``YYYY-MM``, ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS`` for an ISO-shaped cell, else None.

    The result is one of the normalized forms ``timestamp_from_text`` accepts
    without inference; a cell in the right shape that is not a real date
    (``2024-02-30``) is rejected there, not here. A time needs a full date.

    dlt already stores the timestamps it types in UTC; a zoned time reaches
    this function only from a column dlt left as text, and is shifted to UTC
    by dlt's own normalizer so the instant, not the wall-clock digits, is kept.
    """
    if value is None:
        return None
    text = str(value).strip()
    match = _ISO_CELL.match(text)
    if match is None:
        return None
    date, time, zone = match.groups()
    if time and len(date) != 10:
        return None
    if zone:
        return ensure_pendulum_datetime_utc(text).strftime("%Y-%m-%d %H:%M:%S")
    return f"{date} {time}" if time else date


def temporal_cells(row: DltRowData) -> dict[str, str]:
    """``{column: normalized timestamp string}`` for the row's ISO-shaped cells."""
    fk_columns = {fk.get("column", "") for fk in row.foreign_keys}
    cells = {}
    for column, value in row.row_data.items():
        if column in fk_columns:
            continue
        normalized = timestamp_str_for_cell(value)
        if normalized is not None:
            cells[column] = normalized
    return cells
