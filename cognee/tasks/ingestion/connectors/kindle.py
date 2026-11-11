"""Kindle connector for cognee: a ``dlt`` source that reads Kindle clippings, one document per book.

Readers accumulate highlights and notes on Kindle in a single file (`My Clippings.txt`).
This connector parses the file, deduplicates overlapping/extended highlights, attaches notes
to their corresponding highlights, and emits one structured Markdown document per book.
It supports incremental sync, stable book IDs, and tombstone deletions when books are removed.
"""

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)

logger = get_logger("kindle_connector")

# Delimiter used by Amazon Kindle to separate clippings
_CLIPPING_DELIMITER = "=========="

# Regex for Title and Author line: "Title (Author)" or "Title"
_TITLE_AUTHOR_RE = re.compile(r"^(?P<title>.+?)(?:\s*\((?P<author>[^()]+)\))?\s*$")

# Regex to detect clipping type
_HIGHLIGHT_TYPE_RE = re.compile(r"\bhighlight\b", re.IGNORECASE)
_NOTE_TYPE_RE = re.compile(r"\bnote\b", re.IGNORECASE)
_BOOKMARK_TYPE_RE = re.compile(r"\bbookmark\b", re.IGNORECASE)

# Regex to extract location and page
_LOCATION_RE = re.compile(r"(?:location|loc\.?)\s*(\d+)(?:\s*-\s*(\d+))?", re.IGNORECASE)
_PAGE_RE = re.compile(r"page\s*(\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"Added on\s+(.*)$", re.IGNORECASE)


class KindleSourceError(RuntimeError):
    """Base error for Kindle source connector."""


class KindleParseError(KindleSourceError):
    """Raised when My Clippings.txt contains unparseable or malformed content."""


@dataclass
class _HighlightItem:
    loc_start: int
    loc_end: int
    page: int | None
    added_on: str
    text: str
    notes: list[str] = field(default_factory=list)


@dataclass
class _StandaloneNoteItem:
    loc: int
    page: int | None
    added_on: str
    text: str


@dataclass
class _BookAccumulator:
    title: str
    author: str
    highlights: list[_HighlightItem] = field(default_factory=list)
    standalone_notes: list[_StandaloneNoteItem] = field(default_factory=list)


def _book_id(title: str, author: str) -> str:
    """Stable ID for a book based on title and author."""
    slug = hashlib.sha256(f"{title}::{author}".encode()).hexdigest()[:16]
    return f"kindle:{slug}"


def _parse_entry(
    raw_entry: str, entry_index: int
) -> tuple[str, str, str, int, int, int | None, str, str] | None:
    """Parse a single clipping entry.

    Returns:
        (entry_type, title, author, loc_start, loc_end, page, added_on, text)
        or None if the entry is a bookmark or empty.
    """
    lines = [line.rstrip("\r") for line in raw_entry.strip("\r\n").split("\n")]
    # Strip empty leading/trailing lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return None

    if len(lines) < 2:
        raise KindleParseError(
            f"Clipping entry #{entry_index} is malformed (fewer than 2 lines): {raw_entry[:100]!r}"
        )

    title_line = lines[0].strip()
    meta_line = lines[1].strip()
    text = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""

    # Parse title and author
    title_match = _TITLE_AUTHOR_RE.match(title_line)
    if title_match:
        title = title_match.group("title").strip()
        author = title_match.group("author") or ""
        author = author.strip()
    else:
        title = title_line
        author = ""

    # Detect type
    if _HIGHLIGHT_TYPE_RE.search(meta_line):
        entry_type = "highlight"
    elif _NOTE_TYPE_RE.search(meta_line):
        entry_type = "note"
    elif _BOOKMARK_TYPE_RE.search(meta_line):
        # Bookmarks have no text and are ignored
        return None
    else:
        raise KindleParseError(
            f"Clipping entry #{entry_index} has unknown type in metadata: {meta_line!r}"
        )

    # Extract location and page
    loc_match = _LOCATION_RE.search(meta_line)
    page_match = _PAGE_RE.search(meta_line)
    page = int(page_match.group(1)) if page_match else None

    if loc_match:
        loc_start = int(loc_match.group(1))
        loc_end = int(loc_match.group(2)) if loc_match.group(2) else loc_start
    elif page is not None:
        loc_start = page
        loc_end = page
    else:
        raise KindleParseError(
            f"Clipping entry #{entry_index} is missing location and page: {meta_line!r}"
        )

    # Extract date
    date_match = _DATE_RE.search(meta_line)
    added_on = date_match.group(1).strip() if date_match else ""

    return (entry_type, title, author, loc_start, loc_end, page, added_on, text)


def _ranges_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Return True if range [s1, e1] overlaps with [s2, e2]."""
    return max(s1, s2) <= min(e1, e2)


def _add_highlight(book: _BookAccumulator, item: _HighlightItem) -> None:
    """Add a highlight to the book.

    If the highlight's location overlaps with an existing highlight, the newer
    highlight replaces the older one (per Kindle extended highlight behavior),
    preserving any notes attached to the older highlight.
    """
    kept: list[_HighlightItem] = []
    absorbed_notes: list[str] = []

    for existing in book.highlights:
        if _ranges_overlap(existing.loc_start, existing.loc_end, item.loc_start, item.loc_end):
            # Overlap: keep notes attached to previous highlight
            absorbed_notes.extend(existing.notes)
        else:
            kept.append(existing)

    item.notes.extend(absorbed_notes)
    kept.append(item)
    book.highlights = kept


def _attach_note(
    book: _BookAccumulator, loc: int, page: int | None, added_on: str, note_text: str
) -> None:
    """Attach note to the matching highlight, or store as standalone note."""
    if not note_text:
        return

    # Check if note location falls within any highlight range
    matched_highlight: _HighlightItem | None = None
    for highlight in book.highlights:
        if highlight.loc_start <= loc <= highlight.loc_end:
            matched_highlight = highlight
            break

    if matched_highlight is not None:
        matched_highlight.notes.append(note_text)
    else:
        book.standalone_notes.append(
            _StandaloneNoteItem(loc=loc, page=page, added_on=added_on, text=note_text)
        )


def _generate_book_markdown(book: _BookAccumulator) -> str:
    """Generate Markdown document for a book with all its highlights and notes."""
    lines: list[str] = [f"# {book.title}", ""]
    if book.author:
        lines.extend([f"**Author:** {book.author}", ""])

    lines.extend(["## Highlights & Notes", ""])

    # Combine highlights and standalone notes sorted by location
    entries: list[tuple[int, str, Any]] = []
    for h in book.highlights:
        entries.append((h.loc_start, "highlight", h))
    for sn in book.standalone_notes:
        entries.append((sn.loc, "note", sn))

    entries.sort(key=lambda x: x[0])

    for _, kind, entry in entries:
        if kind == "highlight":
            h: _HighlightItem = entry
            loc_str = (
                f"Location {h.loc_start}"
                if h.loc_start == h.loc_end
                else f"Location {h.loc_start}-{h.loc_end}"
            )
            page_str = f" (page {h.page})" if h.page is not None else ""
            lines.append(f"### {loc_str}{page_str}")
            if h.added_on:
                lines.append(f"*Added on {h.added_on}*")
            lines.append("")

            # Quote highlight text
            quoted_text = "\n".join(f"> {line}" for line in h.text.split("\n"))
            lines.extend([quoted_text, ""])

            if h.notes:
                if len(h.notes) == 1:
                    lines.extend([f"**Note:** {h.notes[0]}", ""])
                else:
                    lines.append("**Notes:**")
                    for n in h.notes:
                        lines.append(f"- {n}")
                    lines.append("")
        else:
            sn: _StandaloneNoteItem = entry
            loc_str = f"Location {sn.loc}"
            page_str = f" (page {sn.page})" if sn.page is not None else ""
            lines.append(f"### {loc_str}{page_str} (Note)")
            if sn.added_on:
                lines.append(f"*Added on {sn.added_on}*")
            lines.append("")
            lines.extend([sn.text, ""])

    return "\n".join(lines).strip()


def _parse_clippings(content: str) -> dict[tuple[str, str], _BookAccumulator]:
    """Parse raw clippings text into books with highlights and notes."""
    raw_entries = content.split(_CLIPPING_DELIMITER)
    books: dict[tuple[str, str], _BookAccumulator] = {}

    for idx, raw_entry in enumerate(raw_entries, start=1):
        parsed = _parse_entry(raw_entry, idx)
        if parsed is None:
            continue

        entry_type, title, author, loc_start, loc_end, page, added_on, text = parsed
        book_key = (title, author)
        if book_key not in books:
            books[book_key] = _BookAccumulator(title=title, author=author)

        book = books[book_key]
        if entry_type == "highlight":
            if text:
                item = _HighlightItem(
                    loc_start=loc_start,
                    loc_end=loc_end,
                    page=page,
                    added_on=added_on,
                    text=text,
                )
                _add_highlight(book, item)
        elif entry_type == "note":
            _attach_note(book, loc_start, page, added_on, text)

    return books


def _book_row(title: str, author: str, content: str) -> dict[str, Any]:
    """Format one document row for cognee memory."""
    row: dict[str, Any] = {
        "id": _book_id(title, author),
        "title": title,
        "content": content,
        "url": "",
        "_deleted": False,
    }
    node_set = ["kindle:book"]
    if author:
        clean_author = re.sub(r"[^\w\s-]", "", author).strip()
        if clean_author:
            node_set.append(f"kindle:author:{clean_author}")
    row[NODE_SET_COLUMN] = node_set
    return row


def _iter_rows(
    clippings_path: Path,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield document rows for all books and tombstones for removed books."""
    if not clippings_path.is_file():
        raise KindleSourceError(
            f"The Kindle clippings file {str(clippings_path)!r} does not exist."
        )

    try:
        content = clippings_path.read_text(encoding="utf-8-sig")
    except OSError as err:
        raise KindleSourceError(f"Could not read Kindle clippings file: {err}") from err

    previous_members = set(state.get("members", []))

    if previous_members and not content.strip():
        raise KindleSourceError(
            "The Kindle clippings file is empty, but earlier runs synced books from it; "
            "nothing was synced and nothing was forgotten. Ensure the Kindle file is intact."
        )

    books = _parse_clippings(content)
    present_members = set()

    for (title, author), book in books.items():
        if check_active is not None:
            check_active()

        book_id = _book_id(title, author)
        present_members.add(book_id)

        markdown_doc = _generate_book_markdown(book)
        yield _book_row(title, author, markdown_doc)
        stats["synced"] += 1

    # Tombstones for books that dropped out of the file
    for deleted_id in sorted(previous_members - present_members):
        yield {"id": deleted_id, "_deleted": True}
        stats["deleted"] += 1

    state.update(v=1, members=sorted(present_members))
    logger.info(
        "Kindle sync complete: %d book(s) synced, %d forgotten.",
        stats["synced"],
        stats["deleted"],
    )


def kindle_source(
    *,
    clippings_path: str | os.PathLike[str],
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per book from a Kindle clippings file.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``.

    Args:
        clippings_path: Path to ``My Clippings.txt`` on the Kindle or local storage.
        resource_name: Names the staging table and sync state. Defaults to a hash of the file path.
        check_active: Host authorization checkpoint called before publishing rows.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Kindle sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )

    file_path = Path(clippings_path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"clippings_path must be an existing file: {clippings_path}")

    if resource_name is None:
        resource_name = f"kindle_{hashlib.sha256(str(file_path).encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def kindle_clippings():
        stats.update(synced=0, deleted=0, failed=0)
        rows = _iter_rows(file_path, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = kindle_clippings()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "kindle")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
