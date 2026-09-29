"""BROAD's record store: a dataset's text as SQLite tables that one query answers exactly.

Every document becomes rows. Records (delimited text with a header, markdown tables, JSON
records, an mbox email archive, DLT rows) become one table per set of columns. All other
text becomes the ``lines`` table, one row per non-empty line, with the line's document,
the first line of its document and of its block (so a line keeps the heading it sits
under), and its shape: the line with every value masked, and the values in order
(``v1`` ... ``v8``). A model writes one read-only SELECT over these tables; SQLite
computes the answer.
"""

import csv
import email
import email.utils
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import regex

# Fewer records than this are not trusted to be a table; their text goes to ``lines``.
TABLE_MIN_ROWS = 10
# Share of records that must split into the same number of fields.
TABLE_MIN_AGREEMENT = 0.95
TABLE_SEPARATORS = (",", "\t", "|", ";")
# A column is numeric when at least this share of its non-empty values are numbers.
NUMERIC_SHARE = 0.95
# Values of a line kept as columns v1 ... v8.
LINE_VALUES = 8
# Line shapes are shown only when the text is templated: at most this many distinct
# shapes per line (prose has about one shape per line).
TEMPLATED_SHAPE_SHARE = 0.2
SHOWN_SHAPES = 40
# How the text is laid out, shown to the model: the first lines of a few documents, and
# the short lines that look like headings (a chapter or section title, a name, a label).
PREVIEW_DOCUMENTS = 3
PREVIEW_LINES = 12
HEADING_CHARS = 80
SHOWN_HEADINGS = 10
SHOWN_REPEATS = 8
_HEADING = regex.compile(
    r"^(chapter|book|part|section|act|scene|volume|canto|letter|epilogue|prologue|preface|"
    r"introduction|appendix|article|\d+[.)]?|[ivxlc]+[.)]?)(\b|$)",
    regex.IGNORECASE,
)
# What the model is shown of each table.
SAMPLE_ROWS = 3
TOP_VALUES = 8
TOP_VALUES_MAX_DISTINCT = 50
VALUE_CHARS = 60
# Values matching a word of the question, shown per column (the spelling the data uses).
PROBE_MIN_CHARS = 3
PROBE_VALUES = 5
PROBE_WORDS = 12
# Rows a query may return, and how long it may run.
MAX_RESULT_ROWS = 1_000
MAX_QUERY_SECONDS = 120.0

_MBOX_SEPARATOR = re.compile(r"(?m)^From \S+ .*\n(?=[A-Za-z-]+: )")
_MAIL_BODY_CHARS = 2_000
# A value in a line: a run of two or more capitalised words (a name, in any script with
# case), or a token holding a digit (a number, date, time, id, address or code).
_VALUE = regex.compile(
    r"\b\p{Lu}[\p{L}\p{M}'-]*(?:\s+\p{Lu}[\p{L}\p{M}'-]*)+\b"
    r"|[^\s,;:()\[\]{}\"']*\d[^\s,;:()\[\]{}\"']*"
)
# A number as tables write it: optional sign, thousands separated by commas, a decimal part.
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


class BroadQueryError(ValueError):
    """A query that is not a single read-only SELECT, fails, or runs too long."""


@dataclass
class Table:
    name: str
    columns: list[str]
    rows: list[list[str]]
    foreign_keys: list[str] = field(default_factory=list)


@dataclass
class _Built:
    columns: list[str]
    types: list[str]
    rows: int = 0
    documents: set[str] = field(default_factory=set)
    foreign_keys: set[str] = field(default_factory=set)


def _number(value: str) -> int | float | None:
    """The value as a number, or None. An id with a leading zero ("00123") stays text."""
    cleaned = value.strip()
    if not _NUMBER.fullmatch(cleaned):
        return None
    digits = cleaned.lstrip("+-")
    if len(digits) > 1 and digits[0] == "0" and digits[1] != ".":
        return None
    number = float(cleaned.replace(",", ""))
    return int(number) if "." not in cleaned and number.is_integer() else number


def _column_type(values: list[str]) -> str:
    filled = [v for v in values if v.strip()]
    numbers = [_number(v) for v in filled]
    if not filled or sum(n is not None for n in numbers) < NUMERIC_SHARE * len(filled):
        return "TEXT"
    return "INTEGER" if all(isinstance(n, int) for n in numbers if n is not None) else "REAL"


def _identifier(name: str, taken: set[str]) -> str:
    base = re.sub(r"\W+", "_", name.strip().lower()).strip("_") or "column"
    if base[0].isdigit():
        base = f"t_{base}"
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    taken.add(candidate)
    return candidate


def norm(value: object) -> str | None:
    """Text for comparison: case folded, accents removed, spaces collapsed."""
    if value is None:
        return None
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


def _words(text: str | None, term: str | None) -> int:
    """How many times ``term`` appears in ``text`` as whole words, compared by norm()."""
    if not text or not term:
        return 0
    words = [re.escape(w) for w in (norm(term) or "").split()]
    return len(re.findall(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", norm(text) or ""))


# --- recognising records -----------------------------------------------------------------


def parse_records(name: str, text: str) -> Table | None:
    """The document as a table of records, or None when it is not records."""
    return _json_records(name, text) or _mail_records(name, text) or _delimited(name, text)


def _flatten(record: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in record.items():
        column = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{column}."))
        elif isinstance(value, list):
            flat[column] = " | ".join(
                json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                for v in value
            )
        elif isinstance(value, bool):
            flat[column] = "true" if value else "false"
        else:
            flat[column] = "" if value is None else str(value)
    return flat


def _json_records(name: str, text: str) -> Table | None:
    """A JSON array of objects, an object holding one, or JSON Lines."""
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
        lists = [v for v in data.values() if isinstance(v, list)]
        data = max(lists, key=len) if lists else None
    if not isinstance(data, list) or len(data) < 2 or not all(isinstance(r, dict) for r in data):
        return None
    records = [_flatten(record) for record in data]
    columns = list(dict.fromkeys(key for record in records for key in record))
    return Table(name, columns, [[record.get(c, "") for c in columns] for record in records])


def _mail_records(name: str, text: str) -> Table | None:
    """An mbox email archive, one row per message."""
    starts = [m.start() for m in _MBOX_SEPARATOR.finditer(text)]
    if len(starts) < 2 or starts[0] != 0:
        return None
    rows = []
    for begin, end in zip(starts, [*starts[1:], len(text)]):
        message = email.message_from_string(text[begin:end].split("\n", 1)[1])
        sender, address = email.utils.parseaddr(message.get("From", ""))
        body = next(
            (
                str(part.get_payload())
                for part in message.walk()
                if part.get_content_type() == "text/plain" and not part.is_multipart()
            ),
            "",
        )
        rows.append(
            [
                sender,
                address,
                message.get("To", ""),
                message.get("Date", ""),
                " ".join((message.get("Subject") or "").split()),
                message.get("In-Reply-To", ""),
                body[:_MAIL_BODY_CHARS],
            ]
        )
    columns = ["from_name", "from_address", "to", "date", "subject", "in_reply_to", "body"]
    return Table(name, columns, rows)


def _split(text: str, separator: str) -> list[list[str]]:
    """Records split by the separator; a markdown table's outer pipes and its ---|---
    rule row are dropped."""
    records = []
    for record in csv.reader(io.StringIO(text), delimiter=separator):
        cells = [cell.strip() for cell in record]
        if separator == "|" and len(cells) > 2 and not cells[0] and not cells[-1]:
            cells = cells[1:-1]
        if any(cells) and not all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            records.append(cells)
    return records


def _delimited(name: str, text: str) -> Table | None:
    """Delimited records with a header row, split by the separator most records agree on."""
    best: tuple[float, str] | None = None
    sample = text[:65_536]
    for separator in TABLE_SEPARATORS:
        if sample.count(separator) < TABLE_MIN_ROWS:
            continue
        records = _split(sample, separator)
        if len(records) < 2:
            continue
        width = Counter(len(r) for r in records).most_common(1)[0][0]
        agreement = sum(len(r) == width for r in records) / len(records)
        if width >= 2 and agreement >= TABLE_MIN_AGREEMENT and (not best or agreement > best[0]):
            best = (agreement, separator)
    if best is None:
        return None
    records = _split(text, best[1])
    header, body = records[0], records[1:]
    width = len(header)
    agreement = sum(len(r) == width for r in body) / max(len(body), 1)
    if len(body) < TABLE_MIN_ROWS or agreement < TABLE_MIN_AGREEMENT or len(set(header)) != width:
        return None
    return Table(name, header, [r[:width] + [""] * (width - len(r)) for r in body])


def parse_dlt_row(text: str) -> tuple[str, dict[str, str], list[str]] | None:
    """A DLT row node's text as its table name, its fields and its foreign keys. The text
    is "Table: x", optional "Columns:" and "Foreign Keys:" sections, then "Row Data:" with
    one "key: value" line per field."""
    head = re.match(r"\s*Table:\s*(.+)", text)
    marker = text.find("Row Data:")
    if not head or marker < 0:
        return None
    keys_at = text.find("Foreign Keys:")
    foreign_keys = []
    if 0 <= keys_at < marker:
        foreign_keys = [
            line.strip()[2:].strip()
            for line in text[keys_at:marker].splitlines()[1:]
            if line.strip().startswith("- ")
        ]
    fields: dict[str, str] = {}
    for line in text[marker + len("Row Data:") :].splitlines():
        key, sep, value = line.strip().partition(": ")
        if sep:
            fields[key] = value
        elif line.strip().endswith(":"):
            fields[line.strip()[:-1]] = ""
    return head.group(1).strip(), fields, foreign_keys


def _is_heading(line: str) -> bool:
    """A short line that looks like a heading: in capitals, or opening with a chapter, part,
    section or number word."""
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and (all(c.isupper() for c in letters) or _HEADING.match(line) is not None)


def shape_of(line: str) -> tuple[str, list[str]]:
    """The line with its values masked (@ a name, # anything holding a digit), and the
    values in order."""
    values: list[str] = []

    def mask(match) -> str:
        values.append(match.group(0))
        return "#" if any(c.isdigit() for c in match.group(0)) else "@"

    return " ".join(_VALUE.sub(mask, line).split()), values


_READ_ONLY = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


class RecordStore:
    """A temporary SQLite database of one dataset's records, built document by document."""

    def __init__(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="broad_")
        self.connection = sqlite3.connect(os.path.join(self.directory, "records.db"))
        self.connection.create_function("words", 2, _words, deterministic=True)
        self.connection.create_function("norm", 1, norm, deterministic=True)
        self.tables: dict[str, _Built] = {}
        self.by_columns: dict[tuple[str, ...], str] = {}
        self.names: set[str] = {"lines"}
        self.line_rows = 0
        self.connection.execute(
            "CREATE TABLE lines (document TEXT, title TEXT, block TEXT, line_no INTEGER, "
            "text TEXT, shape TEXT, "
            + ", ".join(f"v{n} TEXT" for n in range(1, LINE_VALUES + 1))
            + ")"
        )

    def close(self) -> None:
        self.connection.close()
        shutil.rmtree(self.directory, ignore_errors=True)

    # --- building --------------------------------------------------------------------

    def add_document(self, name: str, text: str) -> None:
        table = parse_records(name, text)
        if table is None:
            self._add_lines(name, text)
        else:
            self._add_table(table)

    def add_dlt_rows(self, texts: list[str]) -> None:
        by_table: dict[str, tuple[list[dict[str, str]], set[str]]] = {}
        for text in texts:
            parsed = parse_dlt_row(text)
            if parsed:
                rows, keys = by_table.setdefault(parsed[0], ([], set()))
                rows.append(parsed[1])
                keys.update(parsed[2])
        for name, (records, keys) in by_table.items():
            columns = [
                c for c in dict.fromkeys(k for r in records for k in r) if not c.startswith("_dlt")
            ]
            rows = [[r.get(c, "") for c in columns] for r in records]
            self._add_table(Table(name, columns, rows, sorted(keys)))

    def _add_table(self, table: Table) -> None:
        key = tuple(table.columns)
        name = self.by_columns.get(key)
        if name is None:
            # A table is named after its document, without a file extension.
            name = _identifier(re.sub(r"\.\w{1,5}$", "", table.name), self.names)
            types = [_column_type([row[i] for row in table.rows]) for i in range(len(key))]
            taken: set[str] = {"_document"}
            columns = [_identifier(c, taken) for c in table.columns]
            self.connection.execute(
                f'CREATE TABLE "{name}" (_document TEXT, '
                + ", ".join(f'"{c}" {t}' for c, t in zip(columns, types))
                + ")"
            )
            self.tables[name] = _Built(columns=columns, types=types)
            self.by_columns[key] = name
        built = self.tables[name]

        def stored(value: str, kind: str) -> object:
            # A number where the column holds numbers; else the text as written (SQLite
            # keeps it), so a later document's odd value is not lost.
            number = _number(value) if kind != "TEXT" else None
            return number if number is not None else (value if value.strip() else None)

        values = [
            [table.name] + [stored(v, t) for v, t in zip(row, built.types)] for row in table.rows
        ]
        marks = ", ".join("?" * (len(built.columns) + 1))
        self.connection.executemany(f'INSERT INTO "{name}" VALUES ({marks})', values)
        built.rows += len(values)
        built.documents.add(table.name)
        built.foreign_keys.update(table.foreign_keys)

    def _add_lines(self, name: str, text: str) -> None:
        rows = []
        title = block = ""
        for number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line:
                block = ""
                continue
            title = title or line
            block = block or line
            shape, values = shape_of(line)
            padded = (values + [None] * LINE_VALUES)[:LINE_VALUES]
            rows.append([name, title, block, number, line, shape, *padded])
        marks = ", ".join("?" * (6 + LINE_VALUES))
        self.connection.executemany(f"INSERT INTO lines VALUES ({marks})", rows)
        self.line_rows += len(rows)

    # --- what the model is shown -----------------------------------------------------

    def describe(self, question: str) -> str:
        """Every table with its columns, row count, common values, a few rows, and the
        values that contain words of the question (the spelling the data uses)."""
        self.connection.commit()
        words = self._question_words(question)
        parts = []
        for name, built in self.tables.items():
            lines = [
                (
                    f'TABLE "{name}" ({built.rows} rows, from {len(built.documents)} document(s)); '
                    "column _document names the document a row came from"
                )
            ]
            for column, kind in zip(built.columns, built.types):
                lines.append(f'- "{column}" {kind}{self._common(name, column, kind)}')
            if built.foreign_keys:
                lines.append("Foreign keys: " + "; ".join(sorted(built.foreign_keys)))
            lines.append(self._sample(f'SELECT * FROM "{name}" LIMIT {SAMPLE_ROWS}'))
            matches = self._probe(name, built, words)
            if matches:
                lines.append("Values containing words of the question: " + matches)
            parts.append("\n".join(lines))
        if self.line_rows:
            parts.append(self._describe_lines())
        return "\n\n".join(parts)

    def _describe_lines(self) -> str:
        shapes = self.connection.execute("SELECT COUNT(DISTINCT shape) FROM lines").fetchone()[0]
        text = (
            f"TABLE lines ({self.line_rows} rows: every non-empty line of the text documents)\n"
            "- document TEXT: the document's name; title TEXT: its first line; block TEXT: the "
            "first line of the blank-line-separated block the line is in\n"
            "- line_no INTEGER, text TEXT: the line\n"
            "- shape TEXT: the line with each value masked (@ a run of capitalised words such as "
            "a name, # a token holding a digit); v1 ... v8 TEXT: those values in order\n"
        )
        if shapes <= TEMPLATED_SHAPE_SHARE * self.line_rows:
            common = self.connection.execute(
                "SELECT shape, COUNT(*), MIN(text) FROM lines GROUP BY shape "
                f"ORDER BY 2 DESC LIMIT {SHOWN_SHAPES}"
            ).fetchall()
            text += (
                f"The {len(common)} commonest of {shapes} shapes:\n"
                + "\n".join(
                    f"  {count} lines: {shape[:120]}   e.g. {example[:120]}"
                    for shape, count, example in common
                )
                + "\n"
            )
        else:
            text += f"({shapes} different shapes: free text, not templated)\n"
        return text + self._layout() + self._sample(f"SELECT * FROM lines LIMIT {SAMPLE_ROWS}")

    def _layout(self) -> str:
        """The first lines of a few documents, the heading-like lines, and the short lines
        that repeat: how the text is laid out, which no query can tell from counts."""
        out = []
        documents = self.connection.execute(
            f"SELECT DISTINCT document FROM lines ORDER BY document LIMIT {PREVIEW_DOCUMENTS}"
        ).fetchall()
        for (document,) in documents:
            head = self.connection.execute(
                "SELECT line_no, text FROM lines WHERE document = ? ORDER BY line_no LIMIT ?",
                (document, PREVIEW_LINES),
            ).fetchall()
            out.append(
                f"Start of document {document!r}:\n"
                + "\n".join(f"  {n}: {t[: VALUE_CHARS * 2]}" for n, t in head)
            )
        short = self.connection.execute(
            "SELECT line_no, text FROM lines WHERE length(text) <= ? ORDER BY line_no",
            (HEADING_CHARS,),
        ).fetchall()
        headings = [(n, t) for n, t in short if _is_heading(t)]
        if headings:
            shown = headings[:SHOWN_HEADINGS]
            more = (
                f" ... and {len(headings) - len(shown)} more" if len(headings) > len(shown) else ""
            )
            out.append(
                f"Heading-like lines ({len(headings)}; a short line in capitals or starting with "
                "a chapter, part, section or number word): "
                + ", ".join(f"line {n} {t!r}" for n, t in shown)
                + more
            )
        repeats = self.connection.execute(
            "SELECT text, COUNT(*) FROM lines WHERE length(text) <= ? GROUP BY text "
            f"HAVING COUNT(*) > 1 ORDER BY 2 DESC, MIN(line_no) LIMIT {SHOWN_REPEATS}",
            (HEADING_CHARS,),
        ).fetchall()
        distinct_repeated = self.connection.execute(
            "SELECT COUNT(*) FROM (SELECT text FROM lines WHERE length(text) <= ? "
            "GROUP BY text HAVING COUNT(*) > 1)",
            (HEADING_CHARS,),
        ).fetchone()[0]
        if repeats:
            out.append(
                f"Short lines that appear more than once ({distinct_repeated} different texts): "
                + ", ".join(f"{t!r} x{c}" for t, c in repeats)
            )
        return "\n".join(out) + "\n"

    @staticmethod
    def _question_words(question: str) -> list[str]:
        words = regex.findall(r"[\p{L}\p{N}][\p{L}\p{N}'-]*", norm(question) or "")
        longest_first = sorted(
            {w for w in words if len(w) >= PROBE_MIN_CHARS}, key=len, reverse=True
        )
        return longest_first[:PROBE_WORDS]

    def _probe(self, table: str, built: _Built, words: list[str]) -> str:
        """For each text column, the values containing a word of the question."""
        found = []
        for column, kind in zip(built.columns, built.types):
            if kind != "TEXT" or not words:
                continue
            hits: list[str] = []
            for word in words:
                rows = self.connection.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" WHERE norm("{column}") LIKE ? '
                    f"LIMIT {PROBE_VALUES}",
                    (f"%{word}%",),
                ).fetchall()
                for (value,) in rows:
                    shown = str(value)[:VALUE_CHARS]
                    if shown not in hits:
                        hits.append(shown)
            if hits:
                found.append(f'"{column}": ' + ", ".join(repr(h) for h in hits[: PROBE_VALUES * 2]))
        return "; ".join(found)

    def _common(self, table: str, column: str, kind: str) -> str:
        if kind != "TEXT":
            low, high = self.connection.execute(
                f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"'
            ).fetchone()
            return f"; from {low} to {high}"
        distinct = self.connection.execute(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
        ).fetchone()[0]
        if distinct > TOP_VALUES_MAX_DISTINCT:
            return f"; {distinct} different values"
        top = self.connection.execute(
            f'SELECT "{column}", COUNT(*) FROM "{table}" GROUP BY 1 ORDER BY 2 DESC '
            f"LIMIT {TOP_VALUES}"
        ).fetchall()
        return f"; {distinct} different values, e.g. " + ", ".join(
            f"{str(v)[:VALUE_CHARS]!r} ({n})" for v, n in top
        )

    def _sample(self, sql: str) -> str:
        cursor = self.connection.execute(sql)
        header = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return "  rows: " + " || ".join(
            "; ".join(
                f"{h}={str(v)[:VALUE_CHARS]!r}" for h, v in zip(header, row) if v not in (None, "")
            )
            for row in rows
        )

    # --- answering -------------------------------------------------------------------

    def run(self, sql: str) -> tuple[list[str], list[tuple], bool]:
        """Run one read-only query: its columns, up to MAX_RESULT_ROWS rows, and whether
        more rows were left out. Raises BroadQueryError for anything else (SQLite itself
        refuses more than one statement)."""
        started = time.monotonic()

        def too_long() -> int:
            return int(time.monotonic() - started > MAX_QUERY_SECONDS)

        def authorize(action, *_):
            return sqlite3.SQLITE_OK if action in _READ_ONLY else sqlite3.SQLITE_DENY

        self.connection.set_authorizer(authorize)
        self.connection.set_progress_handler(too_long, 10_000)
        try:
            cursor = self.connection.execute(sql.strip().rstrip(";"))
            rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        except (sqlite3.Error, sqlite3.Warning) as error:
            raise BroadQueryError(str(error)) from error
        finally:
            self.connection.set_authorizer(None)
            self.connection.set_progress_handler(None, 0)
        columns = [d[0] for d in cursor.description or []]
        return columns, rows[:MAX_RESULT_ROWS], len(rows) > MAX_RESULT_ROWS
