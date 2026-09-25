"""BROAD's table lane: structured records are counted by code, not read.

A document of records (a CSV export, a TSV, a pipe-separated log, a JSON array of
objects, JSON Lines) is parsed into rows. One LLM call maps the question onto the columns (filters,
a grouping, an aggregate); code then evaluates it over every row, so the answer is
exact and costs the same for ten rows as for a million. A question the columns cannot
answer (one that needs a free-text column understood) is left to the reading path.
"""

import csv
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

# Fewer records than this are read as text: a table this small costs little to read.
TABLE_MIN_ROWS = 20
# Share of records that must split into the same number of fields. A record with more
# fields than the rest keeps its extra separators in its last field (free text).
TABLE_MIN_AGREEMENT = 0.95
TABLE_SEPARATORS = (",", "\t", "|", ";")
# What the query call is shown about each column.
TABLE_TOP_VALUES = 12
TABLE_VALUE_CHARS = 60
TABLE_SAMPLE_ROWS = 3


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]


class TableFilter(BaseModel):
    column: str
    op: Literal[
        "equals",
        "not_equals",
        "contains",
        "not_contains",
        "starts_with",
        "empty",
        "not_empty",
        "greater_than",
        "less_than",
    ]
    value: str = ""


class TableQuery(BaseModel):
    # False when the answer needs a free-text column understood, not matched.
    answerable: bool
    reason: str | None = None
    filters: list[TableFilter] = []
    aggregate: Literal["count_rows", "count_distinct", "sum", "average"] = "count_rows"
    # The column counted distinct, summed or averaged.
    column: str | None = None
    group_by: str | None = None
    # One value of group_by the question asks about ("How many commits did Ann make?").
    target: str | None = None
    list_rows: bool = False


@dataclass
class TableAnswer:
    total: float
    rows: int
    groups: list[tuple[str, float]]
    matched: list[list[str]]
    target_values: list[str]


def parse_table(text: str) -> Table | None:
    """The document as a table, or None when it is not records."""
    return _json_table(text) or _delimited_table(text)


def _json_table(text: str) -> Table | None:
    """A JSON array of objects (or an object holding one), or JSON Lines, as rows: one
    column per field, nested fields flattened, lists joined."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        try:
            data = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        except ValueError:
            return None
    if isinstance(data, dict):
        lists = [value for value in data.values() if isinstance(value, list)]
        data = lists[0] if len(lists) == 1 else None
    if not isinstance(data, list) or len(data) < TABLE_MIN_ROWS:
        return None
    if not all(isinstance(record, dict) for record in data):
        return None
    records = [_flatten(record) for record in data]
    columns = list(dict.fromkeys(key for record in records for key in record))
    return Table(columns=columns, rows=[[record.get(c, "") for c in columns] for record in records])


def _flatten(record: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, list):
            flat[name] = ", ".join(
                json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else _text(v)
                for v in value
            )
        else:
            flat[name] = _text(value)
    return flat


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _delimited_table(text: str) -> Table | None:
    """Delimited records (CSV, TSV, pipe- or semicolon-separated) as rows."""
    best: tuple[float, list[list[str]]] | None = None
    for separator in TABLE_SEPARATORS:
        if text.count(separator) < TABLE_MIN_ROWS * 2:
            continue
        records = [
            [cell.strip() for cell in record]
            for record in csv.reader(io.StringIO(text), delimiter=separator)
            if any(cell.strip() for cell in record)
        ]
        if len(records) < TABLE_MIN_ROWS:
            continue
        width, _ = Counter(len(record) for record in records).most_common(1)[0]
        if width < 3:
            continue
        agreement = sum(1 for record in records if len(record) >= width) / len(records)
        if agreement >= TABLE_MIN_AGREEMENT and (best is None or agreement > best[0]):
            fitted = [
                record[: width - 1] + [separator.join(record[width - 1 :])]
                if len(record) > width
                else record + [""] * (width - len(record))
                for record in records
            ]
            best = (agreement, fitted)
    if best is None:
        return None
    records = best[1]
    # A table has at least two columns whose values vary; templated sentences that happen
    # to share their comma count vary in one place at most.
    varying = sum(1 for i in range(len(records[0])) if len({r[i] for r in records[1:]}) > 1)
    if varying < 2:
        return None
    if _is_header(records):
        return Table(columns=records[0], rows=records[1:])
    return Table(columns=[f"column {i + 1}" for i in range(len(records[0]))], rows=records)


def _is_header(records: list[list[str]]) -> bool:
    """A first record of distinct labels, none a number, none reappearing in its column."""
    first = records[0]
    if len(set(first)) != len(first) or any(
        not cell or _number(cell) is not None for cell in first
    ):
        return False
    later = records[1 : 1 + 200]
    return not any(cell in {row[i] for row in later} for i, cell in enumerate(first))


def _number(value: str) -> float | None:
    cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
    try:
        return float(cleaned) if cleaned and re.search(r"\d", cleaned) else None
    except ValueError:
        return None


def describe(table: Table) -> str:
    """Columns with their commonest values and a few whole rows, for the query call."""
    lines = [f"{len(table.rows)} rows. Columns:"]
    for i, name in enumerate(table.columns):
        values = Counter(row[i] for row in table.rows)
        common = ", ".join(
            f"{value[:TABLE_VALUE_CHARS]!r} ({count})"
            for value, count in values.most_common(TABLE_TOP_VALUES)
        )
        lines.append(f"- {name!r}: {len(values)} different values; most common: {common}")
    lines.append("First rows:")
    for row in table.rows[:TABLE_SAMPLE_ROWS]:
        lines.append(
            "  " + " | ".join(f"{c}={v[:TABLE_VALUE_CHARS]!r}" for c, v in zip(table.columns, row))
        )
    return "\n".join(lines)


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


def _keeps(row: list[str], column: int, rule: TableFilter) -> bool:
    cell, value = _fold(row[column]), _fold(rule.value)
    if rule.op == "equals":
        return cell == value
    if rule.op == "not_equals":
        return cell != value
    if rule.op == "contains":
        return value in cell
    if rule.op == "not_contains":
        return value not in cell
    if rule.op == "starts_with":
        return cell.startswith(value)
    if rule.op == "empty":
        return not cell
    if rule.op == "not_empty":
        return bool(cell)
    number, limit = _number(row[column]), _number(rule.value)
    if number is None or limit is None:
        return False
    return number > limit if rule.op == "greater_than" else number < limit


def run_query(table: Table, query: TableQuery) -> TableAnswer:
    """Evaluate the query over every row. Raises KeyError for a column the table lacks."""
    index = {name: i for i, name in enumerate(table.columns)}

    def column(name: str | None) -> int:
        if name not in index:
            raise KeyError(f"BROAD table query names a column the table lacks: {name!r}")
        return index[name]

    rows = table.rows
    for rule in query.filters:
        at = column(rule.column)
        rows = [row for row in rows if _keeps(row, at, rule)]

    target_values: list[str] = []
    if query.target:
        at = column(query.group_by)
        wanted = _fold(query.target)
        target_values = sorted({row[at] for row in rows if _fold(row[at]) == wanted})
        if not target_values:  # a partial name: "Roman" for "Roman Shkarin"
            pattern = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
            target_values = sorted({row[at] for row in rows if pattern.search(_fold(row[at]))})
        rows = [row for row in rows if row[at] in target_values]

    def measure(selected: list[list[str]]) -> float:
        if query.aggregate == "count_rows":
            return len(selected)
        at = column(query.column)
        if query.aggregate == "count_distinct":
            return len({_fold(row[at]) for row in selected if row[at]})
        amounts = [n for n in (_number(row[at]) for row in selected) if n is not None]
        if query.aggregate == "sum":
            return sum(amounts)
        return sum(amounts) / len(amounts) if amounts else 0

    groups: list[tuple[str, float]] = []
    if query.group_by and not query.target:
        at = column(query.group_by)
        members: dict[str, list[list[str]]] = {}
        spellings: dict[str, Counter] = {}
        for row in rows:
            key = _fold(row[at])
            if key:
                members.setdefault(key, []).append(row)
                spellings.setdefault(key, Counter())[row[at]] += 1
        groups = sorted(
            ((spellings[key].most_common(1)[0][0], measure(m)) for key, m in members.items()),
            key=lambda pair: -pair[1],
        )
    return TableAnswer(
        total=measure(rows),
        rows=len(table.rows),
        groups=groups,
        matched=rows,
        target_values=target_values,
    )


def render_row(table: Table, row: list[str]) -> str:
    return "; ".join(f"{c}: {v[:TABLE_VALUE_CHARS]}" for c, v in zip(table.columns, row) if v)
