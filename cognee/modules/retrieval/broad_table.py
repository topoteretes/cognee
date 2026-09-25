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
# A column with at most this many different values shows all of them (row types, levels).
TABLE_ALL_VALUES = 60
TABLE_VALUE_CHARS = 60
TABLE_SAMPLE_ROWS = 3
# Joins the values of a cell that holds several (a JSON list, a line naming three blocks).
MULTI = "\x1f"


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
            flat[name] = MULTI.join(
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


def _values(cell: str) -> list[str]:
    return cell.split(MULTI) if MULTI in cell else [cell]


def describe(table: Table) -> str:
    """Columns with their commonest values and a few whole rows, for the query call."""
    lines = [f"{len(table.rows)} rows. Columns:"]
    for i, name in enumerate(table.columns):
        values = Counter(value for row in table.rows for value in _values(row[i]))
        shown = len(values) if len(values) <= TABLE_ALL_VALUES else TABLE_TOP_VALUES
        common = ", ".join(
            f"{value[:TABLE_VALUE_CHARS]!r} ({count})" for value, count in values.most_common(shown)
        )
        label = "values" if shown == len(values) else "most common"
        lines.append(f"- {name!r}: {len(values)} different values; {label}: {common}")
    lines.append("First rows:")
    for row in table.rows[:TABLE_SAMPLE_ROWS]:
        lines.append(
            "  "
            + " | ".join(
                f"{c}={v.replace(MULTI, ', ')[:TABLE_VALUE_CHARS]!r}"
                for c, v in zip(table.columns, row)
            )
        )
    return "\n".join(lines)


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


def _keeps(row: list[str], column: int, rule: TableFilter) -> bool:
    """A cell with several values meets a positive rule when any value does, and a
    negative rule ("not ...") when every value does."""
    cells, value = [_fold(v) for v in _values(row[column])], _fold(rule.value)
    if rule.op == "equals":
        return value in cells
    if rule.op == "not_equals":
        return value not in cells
    if rule.op == "contains":
        return any(value in cell for cell in cells)
    if rule.op == "not_contains":
        return all(value not in cell for cell in cells)
    if rule.op == "starts_with":
        return any(cell.startswith(value) for cell in cells)
    if rule.op == "empty":
        return not any(cells)
    if rule.op == "not_empty":
        return any(cells)
    number, limit = _number(_values(row[column])[0]), _number(rule.value)
    if number is None or limit is None:
        return False
    return number > limit if rule.op == "greater_than" else number < limit


def run_query(table: Table, query: TableQuery) -> TableAnswer:
    """Evaluate the query over every row. Raises KeyError for a column the table lacks."""
    if query.group_by and query.aggregate == "count_distinct" and query.column == query.group_by:
        # Different values of the column being grouped by are one per group: the question
        # ("which IP made the most requests") counts rows per value.
        query = query.model_copy(update={"aggregate": "count_rows", "column": None})
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
        present = {v for row in rows for v in _values(row[at])}
        target_values = sorted(v for v in present if _fold(v) == wanted)
        if not target_values:  # a partial name: "Roman" for "Roman Shkarin"
            pattern = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
            target_values = sorted(v for v in present if pattern.search(_fold(v)))
        rows = [row for row in rows if set(_values(row[at])) & set(target_values)]

    def measure(selected: list[list[str]]) -> float:
        if query.aggregate == "count_rows":
            return len(selected)
        at = column(query.column)
        if query.aggregate == "count_distinct":
            return len({_fold(v) for row in selected for v in _values(row[at]) if v})
        amounts = [n for n in (_number(_values(row[at])[0]) for row in selected) if n is not None]
        if query.aggregate == "sum":
            return sum(amounts)
        return sum(amounts) / len(amounts) if amounts else 0

    groups: list[tuple[str, float]] = []
    if query.group_by and not query.target:
        at = column(query.group_by)
        members: dict[str, list[list[str]]] = {}
        spellings: dict[str, Counter] = {}
        for row in rows:
            for value in _values(row[at]):
                key = _fold(value)
                if key:
                    members.setdefault(key, []).append(row)
                    spellings.setdefault(key, Counter())[value] += 1
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
    return "; ".join(
        f"{c}: {v.replace(MULTI, ', ')[:TABLE_VALUE_CHARS]}"
        for c, v in zip(table.columns, row)
        if v
    )


# --- line shapes: records that are lines of text, not delimited ----------------------
#
# A log or a templated report writes each record as a line whose wording repeats and
# whose values change. Masking the values (dates, times, numbers, identifiers, names,
# paths) leaves the line's shape; a corpus of records has few shapes for many lines.
# The shapes show a model every form the lines take; for each question it writes
# regular expressions that select the lines and capture the value, and code runs them
# over every line. Coverage is code's, so no line of an overlooked form is dropped.

# Tried only when lines outnumber shapes this many times over, and the shapes shown to
# the model cover this share of lines; prose has about one shape per line.
SHAPE_MIN_COMPRESSION = 5
SHAPE_MIN_COVERAGE = 0.9
# Shapes shown to the model at most, commonest first.
SHAPE_MAX_SHOWN = 300

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
_WEEKDAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*"
# Earlier kinds win where two could match at one position.
_SLOT = re.compile(
    "|".join(
        f"(?P<{kind}>{pattern})"
        for kind, pattern in (
            (
                # A UUID, or a long hexadecimal hash (a request, container or commit id).
                "uuid",
                (
                    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
                    r"|\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{12,}\b"
                ),
            ),
            # A file path or URL path of two or more segments.
            ("path", r"(?:[A-Za-z]:)?(?:[\\/][^\s\\/:*?\"<>|,;()\[\]]+){2,}[\\/]?"),
            (
                "date",
                (
                    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b"
                    rf"|\b\d{{1,2}} {_MONTH} \d{{4}}\b|\b(?:{_WEEKDAY} +)?{_MONTH} +\d{{1,2}}\b"
                ),
            ),
            ("time", r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\b"),
            ("dotted", r"\b\d+(?:\.\d+){2,}\b"),
            ("host", r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+){2,}\b"),
            (
                "code",
                (
                    r"\b[A-Za-z][A-Za-z0-9]*[_-]-?\d[\w-]*\b|\b0x[0-9a-fA-F]+\b"
                    r"|\b[A-Za-z]+\d+[A-Za-z0-9]*\b"
                ),
            ),
            ("name", r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),
            ("num", r"(?<!\w)-?\d[\d,]*(?:\.\d+)?"),
        )
    )
)
_SYMBOL = {
    "uuid": "%",
    "path": "/",
    "date": "$",
    "time": "^",
    "dotted": "~",
    "host": "~",
    "code": "%",
    "name": "@",
    "num": "#",
}


@dataclass
class Line:
    shape: str
    slots: list[tuple[str, str]]  # (kind, value) in line order
    text: str


def shape_of(text: str) -> Line:
    slots: list[tuple[str, str]] = []

    def mask(match: re.Match) -> str:
        kind = match.lastgroup or "num"
        slots.append((kind, match.group(0)))
        return _SYMBOL[kind]

    shape = " ".join(_SLOT.sub(mask, text).split())
    return Line(shape=shape, slots=slots, text=text)


def shaped_lines(text: str) -> tuple[list[Line], list[str]] | None:
    """Every non-empty line with its shape, and the shapes to label (commonest first);
    None when the lines are not records (too many shapes for the lines)."""
    lines = [shape_of(line) for line in text.splitlines() if line.strip()]
    if len(lines) < TABLE_MIN_ROWS:
        return None
    counts = Counter(line.shape for line in lines)
    if len(lines) < SHAPE_MIN_COMPRESSION * len(counts):
        return None
    shapes = [shape for shape, _ in counts.most_common(SHAPE_MAX_SHOWN)]
    covered = sum(counts[shape] for shape in shapes)
    if covered < SHAPE_MIN_COVERAGE * len(lines):
        return None
    return lines, shapes


class LineQuery(BaseModel):
    # False when the lines do not hold what the question asks about.
    answerable: bool
    reason: str | None = None
    # A line counts when this Python regular expression matches it.
    line_regex: str = ""
    # A counted line is left out when this one matches it.
    exclude_regex: str | None = None
    # One capture group: the value counted distinct, grouped by, summed or averaged. Every
    # match on a line is a value of it (a line naming three blocks has three).
    value_regex: str | None = None
    aggregate: Literal["count_rows", "count_distinct", "sum", "average"] = "count_rows"
    # Break the count down by the captured values ("who / which has the most").
    group: bool = False
    # One captured value the question asks about ("How many from 10.0.0.5?").
    target: str | None = None
    list_rows: bool = False


def describe_shapes(shapes: list[str], lines: list[Line]) -> str:
    """Each shape with its line count, one real line, and its slots numbered, for the
    pointing call."""
    example: dict[str, Line] = {}
    counts: Counter = Counter()
    for line in lines:
        example.setdefault(line.shape, line)
        counts[line.shape] += 1
    rows = []
    for index, shape in enumerate(shapes):
        line = example[shape]
        slots = ", ".join(
            f"{n}: {_SYMBOL[kind]} {value[:TABLE_VALUE_CHARS]!r}"
            for n, (kind, value) in enumerate(line.slots)
        )
        rows.append(
            f"{index}. ({counts[shape]} lines) {shape[:220]}\n"
            f"   e.g. {line.text[:220]}\n   slots: [{slots}]"
        )
    return "\n".join(rows)


def matched_table(lines: list[Line], query: LineQuery) -> Table:
    """Every line the query's expressions select, as a table of (captured values, line).
    Raises re.error for an expression that does not compile."""
    keep = re.compile(query.line_regex)
    drop = re.compile(query.exclude_regex) if query.exclude_regex else None
    grab = re.compile(query.value_regex) if query.value_regex else None
    if grab is not None and grab.groups != 1:
        raise re.error("value_regex must have exactly one capture group")
    rows = []
    for line in lines:
        if not keep.search(line.text) or (drop is not None and drop.search(line.text)):
            continue
        values = grab.findall(line.text) if grab is not None else []
        rows.append([MULTI.join(v for v in values if v), line.text])
    return Table(columns=["value", "line"], rows=rows)
