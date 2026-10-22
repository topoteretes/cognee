"""Shape one variant's ticket-proposal tables before the workflow files them.

Each analysis variant writes two files: the full working report (uploaded as
an artifact) and a short issue file. This script is the contract for the
issue file, enforced outside the model:

- the file is two markdown tables with the same proposals: ``## Simply put``
  (``# | Problem | Fix``, plain words) and ``## Details``
  (``# | Evidence | Root cause | Fix shape``). A wrong heading or header, an
  empty cell, text outside the tables, or rows that do not line up fails the
  run;
- every cell is one short line: ``<br>`` is flattened and anything past
  ``MAX_CELL_CHARS`` is cut at a word boundary;
- at most ``MAX_PROPOSALS`` proposals and ``MAX_ISSUE_BODY_CHARS`` characters:
  trailing proposals are dropped from both tables with a note;
- a footer links the run so the full report stays one click away.

Usage: ``python issue_body.py <issue.md> <body-out.md>``. Exits 1 with an
``ISSUE FORMAT:`` message when the file breaks the contract.
"""

import os
import re
import sys
from pathlib import Path

MAX_ISSUE_BODY_CHARS = int(os.getenv("MAX_ISSUE_BODY_CHARS", "7000"))
MAX_PROPOSALS = int(os.getenv("MAX_PROPOSALS", "3"))
MAX_CELL_CHARS = int(os.getenv("MAX_CELL_CHARS", "140"))
TABLES = (
    ("## Simply put", ("#", "Problem", "Fix")),
    ("## Details", ("#", "Evidence", "Root cause", "Fix shape")),
)
TRIM_NOTE = "\n\n_Further proposals were cut by the issue length cap; see the run report._"

Table = list[list[str]]


class IssueFormatError(ValueError):
    """The issue file does not follow the contract the prompt asks for."""


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _header(columns: tuple[str, ...]) -> str:
    return "| " + " | ".join(columns) + " |"


def clip_cell(cell: str, limit: int = MAX_CELL_CHARS) -> tuple[str, bool]:
    """Flatten a cell to one line and cut it at a word boundary under ``limit``."""
    flat = " ".join(re.split(r"<br\s*/?>|\s+", cell)).strip()
    if len(flat) <= limit:
        return flat, flat != cell
    cut = flat.rfind(" ", 0, limit - 1)
    return flat[: cut if cut > limit // 2 else limit - 1].rstrip(" ,;:") + "…", True


def _parse_one(lines: list[str], columns: tuple[str, ...], name: str) -> tuple[Table, list[str]]:
    """Parse one table off the front of ``lines``; return its rows and the rest."""
    if len(lines) < 3:
        raise IssueFormatError(f"{name} must be a table with at least one proposal row")
    if _cells(lines[0]) != list(columns):
        raise IssueFormatError(f"{name} header must be exactly {_header(columns)!r}")
    if not all(cell and set(cell) <= set(":-") for cell in _cells(lines[1])):
        raise IssueFormatError(f"{name} is missing the header separator line")
    rows: Table = []
    rest = lines[2:]
    while rest and rest[0].lstrip().startswith("|"):
        cells = _cells(rest.pop(0))
        if len(cells) != len(columns):
            raise IssueFormatError(
                f"{name} row {len(rows) + 1} has {len(cells)} cells, expected {len(columns)}"
            )
        if not all(cells):
            raise IssueFormatError(f"{name} row {len(rows) + 1} has an empty cell")
        rows.append(cells)
    if not rows:
        raise IssueFormatError(f"{name} has no proposal rows")
    return rows, rest


def parse_tables(body: str) -> list[Table]:
    """Validate the body as the two proposal tables and return their rows."""
    lines = [line for line in body.splitlines() if line.strip()]
    tables: list[Table] = []
    for heading, columns in TABLES:
        if not lines or lines[0].strip() != heading:
            found = lines[0].strip()[:60] if lines else "end of file"
            raise IssueFormatError(f"expected heading {heading!r}, found {found!r}")
        rows, lines = _parse_one(lines[1:], columns, heading)
        tables.append(rows)
    if lines:
        raise IssueFormatError(f"text outside the tables: {lines[0].strip()[:60]!r}")
    numbers = [[row[0] for row in table] for table in tables]
    if any(n != numbers[0] for n in numbers[1:]):
        raise IssueFormatError(f"the two tables list different proposals: {numbers}")
    return tables


def render(tables: list[Table]) -> str:
    parts = []
    for (heading, columns), rows in zip(TABLES, tables):
        separator = "|" + "---|" * len(columns)
        body = [_header(columns), separator, *("| " + " | ".join(row) + " |" for row in rows)]
        parts.append(heading + "\n\n" + "\n".join(body))
    return "\n\n".join(parts)


def trim(
    tables: list[Table], limit: int = MAX_ISSUE_BODY_CHARS, cell_limit: int = MAX_CELL_CHARS
) -> tuple[str, bool]:
    """Clip every cell to one line and drop trailing proposals past the caps.

    Returns ``(body, cut)`` where ``cut`` is true when anything was clipped or
    dropped; the note is appended only when whole proposals were dropped.
    """
    total = len(tables[0])
    kept: list[Table] = []
    clipped_any = False
    for rows in tables:
        kept_rows: Table = []
        for row in rows[:MAX_PROPOSALS]:
            clipped = [clip_cell(cell, cell_limit) for cell in row]
            clipped_any = clipped_any or any(changed for _, changed in clipped)
            kept_rows.append([cell for cell, _ in clipped])
        kept.append(kept_rows)
    while len(kept[0]) > 1 and len(render(kept)) + len(TRIM_NOTE) > limit:
        for rows in kept:
            rows.pop()
    body = render(kept)
    dropped = len(kept[0]) < total
    return (body + TRIM_NOTE if dropped else body), (dropped or clipped_any)


def build(
    text: str,
    run_url: str | None,
    limit: int = MAX_ISSUE_BODY_CHARS,
    cell_limit: int = MAX_CELL_CHARS,
) -> tuple[str, bool]:
    """Return ``(body, trimmed)`` for one variant's section of the issue."""
    body, trimmed = trim(parse_tables(text), limit, cell_limit)
    if run_url:
        body += f"\n\n_Full report, digest and 'Not filed' list: [workflow run]({run_url})._"
    return body, trimmed


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        sys.exit("usage: issue_body.py <issue.md> <body-out.md>")
    source, target = Path(argv[1]), Path(argv[2])
    try:
        body, trimmed = build(source.read_text(), os.getenv("RUN_URL"))
    except IssueFormatError as error:
        sys.exit(f"ISSUE FORMAT: {error}")
    target.write_text(body + "\n")
    if trimmed:
        print(
            f"::warning::proposal tables were cut to {MAX_PROPOSALS} proposals, "
            f"{MAX_CELL_CHARS} chars per cell, {MAX_ISSUE_BODY_CHARS} chars total"
        )
    print(f"body: {len(body)} chars")


if __name__ == "__main__":
    main(sys.argv)
