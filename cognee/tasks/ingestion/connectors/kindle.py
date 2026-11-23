"""Kindle highlights connector for cognee: ``My Clippings.txt`` as one document per book.

Years of highlights and notes sit in one plain-text file on the Kindle, unreachable by
meaning and unconnected across books. This source parses the clippings file and publishes
one markdown document per book — every highlight and note in location order, bookmarks
dropped — with a content-hash id, so an unchanged book is not re-cognified and a book whose
clippings were removed is forgotten on the next sync. When Kindle appends an extended
version of a highlight, the entry that overlaps it is dropped and only the newest text is
kept. See ``kindle.md`` for the entry format, dedup and the failure model.
"""

import hashlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
)

logger = get_logger("kindle_connector")

CLIPPINGS_FILE_NAME = "My Clippings.txt"
# Every entry ends with a line of ten or more equals signs.
_ENTRY_SEPARATOR = re.compile(r"={10,}\s*(?:\r?\n|\Z)")
# The first line of an entry: "Book Title (Author Name)". The author is optional.
_TITLE_LINE = re.compile(r"\A(?P<title>.+?)(?:\s*\((?P<author>[^()].*)\))?\s*\Z")
# The second line: "- Your Highlight on page 55 | location 558-560 | Added on ..."
# Kindle writes the location right after "on" when there is no page ("Your Note on
# location 559") and behind a pipe when there is one. The page and the location are
# optional; the type is not.
_META_LINE = re.compile(
    r"\A-\s*Your\s+(?P<kind>Highlight|Note|Bookmark)\s+on"
    r"(?:\s+page\s+(?P<page>\d+))?"
    r"(?:\s+(?:\|\s*)?location\s+(?P<loc>\d+(?:-\d+)?))?"
    r"(?:\s*\|\s*Added on\s+(?P<added>.+?))?\s*\Z",
    re.IGNORECASE,
)
_ADDED_FORMATS = (
    "%A, %B %d, %Y %I:%M:%S %p",
    "%A, %B %d, %Y %H:%M:%S",
    "%B %d, %Y %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
)


class KindleSourceError(RuntimeError):
    """Base error. Messages carry entry positions, never the clipped text."""


class KindleClippingsUnavailableError(KindleSourceError):
    """The clippings file is gone at run time."""


class KindleClippingsParseError(KindleSourceError):
    """The clippings file is malformed; the run stops before anything is forgotten."""


@dataclass(frozen=True)
class _Clip:
    kind: str  # Highlight, Note or Bookmark
    title: str
    author: str | None
    page: int | None
    location_start: int | None
    location_end: int | None
    added: datetime | None
    text: str


def _parse_added(raw: str | None) -> datetime | None:
    """The entry's timestamp, or None when the metadata line carries no date.

    A date that does not match a known Kindle export format is a malformed entry: dedup
    of extended highlights compares these timestamps, so silently ignoring one could keep
    an outdated text.
    """
    if raw is None or not raw.strip():
        return None
    for fmt in _ADDED_FORMATS:
        try:
            # Kindle writes a local wall-clock time with no zone; UTC is attached so
            # the values compare consistently. The order is what matters, not the zone.
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise KindleClippingsParseError(f"unparseable 'Added on' timestamp: {raw.strip()[:60]}")


def _parse_entry(block: str) -> _Clip:
    """One clippings entry: a title/author line, a metadata line, then the clipped text."""
    lines = [line.strip("\r") for line in block.strip("\r\n").split("\n")]
    lines = [line for line in lines if line.strip()] or [""]
    title_match = _TITLE_LINE.match(lines[0].strip())
    if title_match is None:
        raise KindleClippingsParseError(f"entry has no title line: {lines[0][:60]}")
    meta_match = _META_LINE.match(lines[1].strip()) if len(lines) > 1 else None
    if meta_match is None:
        raise KindleClippingsParseError(
            f"entry has no metadata line (expected '- Your Highlight/Note/Bookmark on ...'): "
            f"{lines[1][:60] if len(lines) > 1 else '(none)'}"
        )
    location = meta_match.group("loc")
    if location:
        start, _, end = location.partition("-")
        location_start, location_end = int(start), int(end if end else start)
    else:
        location_start = location_end = None
    text = "\n".join(lines[2:]).strip()
    return _Clip(
        kind=meta_match.group("kind"),
        title=title_match.group("title").strip(),
        author=title_match.group("author"),
        page=int(meta_match.group("page")) if meta_match.group("page") else None,
        location_start=location_start,
        location_end=location_end,
        added=_parse_added(meta_match.group("added")),
        text=text,
    )


def _parse_clippings(text: str) -> list[_Clip]:
    """Every clip of the file, in file order. An empty file parses to no clips at all."""
    return [_parse_entry(block) for block in _ENTRY_SEPARATOR.split(text) if block.strip()]


def _overlaps(clip: _Clip, other: _Clip) -> bool:
    """Whether two clips' location ranges intersect. A clip without a location never overlaps."""
    if clip.location_start is None or other.location_start is None:
        return False
    return clip.location_start <= other.location_end and other.location_start <= clip.location_end


def _newest_of(clip: _Clip, other: _Clip) -> _Clip:
    """The newer of two clips: a later timestamp wins, file order breaks a tie."""
    if clip.added is None:
        return other
    if other.added is None:
        return clip
    return other if other.added > clip.added else clip


def _dedup_highlights(clips: list[_Clip]) -> list[_Clip]:
    """Drop every clip that an overlapping newer one replaces; keep the file's order.

    Kindle appends an extended highlight instead of updating it: where location ranges
    overlap in the same book, only the newest entry is kept. Highlights, notes and
    bookmarks are deduped among themselves; a note inside a highlight's range is a
    different kind and survives.
    """
    kept: list[_Clip] = []
    for clip in clips:
        for index, other in enumerate(kept):
            if other.kind == clip.kind and _overlaps(clip, other):
                kept[index] = _newest_of(clip, other)
                break
        else:
            kept.append(clip)
    return kept


def _book_document(title: str, author: str | None, clips: list[_Clip]) -> tuple[str, str]:
    """One book's title and markdown body: its clips in location order, bookmarks left out.

    A note is attached to the highlight whose location range contains it, rendered under
    it as a quote; a note with no containing highlight stands on its own line.
    """
    ordered = sorted(
        (clip for clip in clips if clip.kind != "Bookmark"),
        key=lambda clip: (
            clip.location_start if clip.location_start is not None else 0,
            clip.added or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    highlights = [clip for clip in ordered if clip.kind == "Highlight"]
    lines = [f"# {title}"]
    if author:
        lines.append(f"Author: {author}")
    for clip in ordered:
        where = []
        if clip.location_start is not None:
            where.append(
                f"location {clip.location_start}-{clip.location_end}"
                if clip.location_end != clip.location_start
                else f"location {clip.location_start}"
            )
        if clip.page is not None:
            where.append(f"page {clip.page}")
        if clip.added is not None:
            where.append(clip.added.strftime("%Y-%m-%d"))
        prefix = f" ({', '.join(where)})" if where else ""
        if clip.kind == "Highlight":
            attached = [
                note
                for note in ordered
                if note.kind == "Note"
                and note.location_start is not None
                and clip.location_start is not None
                and clip.location_start <= note.location_start <= clip.location_end
            ]
            lines.append(f"\n> {clip.text}{prefix}")
            for note in attached:
                lines.append(f">> Note: {note.text}")
        elif clip.kind == "Note" and not any(
            highlight.location_start is not None
            and clip.location_start is not None
            and highlight.location_start <= clip.location_start <= highlight.location_end
            for highlight in highlights
        ):
            lines.append(f"\n> Note: {clip.text}{prefix}")
    return title, "\n".join(lines).strip()


def _book_rows(text: str) -> list[dict[str, Any]]:
    """One row per book of the clippings file, keyed by the book's content hash."""
    clips = _parse_clippings(text)
    books: dict[tuple[str, str | None], list[_Clip]] = {}
    for clip in clips:
        books.setdefault((clip.title, clip.author), []).append(clip)

    rows: list[dict[str, Any]] = []
    for (title, author), book_clips in sorted(books.items(), key=lambda item: item[0][0]):
        kept = _dedup_highlights(book_clips)
        book_title, body = _book_document(title, author, kept)
        row: dict[str, Any] = {
            "id": "kindle:" + hashlib.sha256(f"{title}|{author}\n{body}".encode()).hexdigest()[:12],
            "title": book_title,
            "content": body,
            "url": "",
            "_deleted": False,
        }
        rows.append(row)
    return rows


def _read_clippings(path: Path) -> str:
    """The clippings text. Kindle writes UTF-8; anything else is refused loudly."""
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise KindleClippingsParseError(
            "the clippings file is not UTF-8; convert it before syncing"
        ) from error
    except OSError as error:
        raise KindleClippingsUnavailableError(
            f"the clippings file could not be read ({type(error).__name__}); nothing was "
            "synced and nothing was forgotten."
        ) from None


def _iter_rows(
    clippings: Path,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one row per book, then a tombstone for every book no longer in the file.

    The whole file is parsed and every row built before the first yield, so a malformed
    entry stops the run with nothing published and nothing forgotten. The state is written
    after the last row, so a run that raises keeps the old one. A file emptied on purpose
    forgets every book, like the clippings deletion it was.
    """
    if not clippings.is_file():
        raise KindleClippingsUnavailableError(
            "The clippings file is not available; nothing was synced and nothing was "
            "forgotten. Check that the Kindle is mounted or point clippings_path at the "
            "new location."
        )
    if check_active is not None:
        check_active()
    rows = _book_rows(_read_clippings(clippings))
    previous = set(state.get("members", []))
    present = {row["id"] for row in rows}
    for row in rows:
        yield row
        # After the yield: a row that check_active stopped before it was published is not counted.
        stats["synced"] += 1
    for book_id in sorted(previous - present):
        yield {"id": book_id, "_deleted": True}
        stats["deleted"] += 1
    state.update(v=1, members=sorted(present))
    logger.info(
        "Kindle: %d book(s) synced, %d forgotten.",
        stats["synced"],
        stats["deleted"],
    )


def kindle_source(
    *,
    clippings_path: str | Path,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per book of a Kindle clippings file.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. A run reads the whole file; a book whose highlights did not change
    makes the same row id, so nothing is re-cognified. A book whose clippings were removed
    is forgotten on the next run.

    Args:
        clippings_path: The ``My Clippings.txt`` file, or a folder that contains it (the
            mounted Kindle's root or its ``documents`` folder). A ``ValueError`` if no
            clippings file is found. A run raises if the file has disappeared by then.
        resource_name: Names the staging table and the sync state. One name is derived from
            the resolved path; pass a name to pin it across a moved mount point.
        check_active: Host authorization checkpoint, called before the file is read and
            before each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Kindle sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    root = Path(clippings_path).expanduser().resolve()
    if root.is_dir():
        nested = root / "documents" / CLIPPINGS_FILE_NAME
        clippings = nested if nested.is_file() else root / CLIPPINGS_FILE_NAME
    else:
        clippings = root
    if not clippings.is_file():
        raise ValueError(
            f"clippings_path must be a {CLIPPINGS_FILE_NAME!r} file or a folder containing "
            "one (the mounted Kindle's root or its documents folder)."
        )
    if resource_name is None:
        resource_name = f"kindle_{hashlib.sha256(str(clippings).encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def kindle_books():
        stats.update(synced=0, deleted=0)
        rows = _iter_rows(clippings, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = kindle_books()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "kindle")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
