"""Subtitles connector for cognee: ``.srt`` and ``.vtt`` transcript files as readable documents.

Meeting recordings, lectures and talks land on disk as subtitle files (Zoom, Teams, Google
Meet, OBS + Whisper). ``cognee.add()`` on the raw file stores cue numbers, timing lines and
markup as graph noise and splits sentences mid-cue. This source parses SRT and WebVTT, strips
the markup, merges consecutive cues into readable paragraphs that keep who said what and when,
and publishes one document per file with a content-hash id, so only changed files are
re-cognified and a file removed from the folder is forgotten on the next sync. See
``subtitles.md`` for what is read, merging and the failure model.
"""

import hashlib
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

logger = get_logger("subtitles_connector")

SUBTITLE_SUFFIXES = (".srt", ".vtt")
# A paragraph absorbs the next cue when it has no more than this many seconds of silence
# before it, and the speaker did not change.
DEFAULT_MIN_GAP_SECONDS = 3.0

# An SRT or WebVTT cue timing line. The hours part is optional in WebVTT ("MM:SS.mmm");
# SRT uses a comma for the milliseconds, WebVTT a dot. Cue settings may follow the arrow.
_TIMING = re.compile(
    r"\A(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)
# WebVTT blocks that carry no speech. A NOTE runs to the end of its block; a STYLE or
# REGION block's body is css-like lines that never contain a timing arrow.
_SKIP_BLOCK = re.compile(r"\A(NOTE|STYLE|REGION)\b")
# Inline markup: voice tags (<v Name>, <v Name (accent)>), formatting (<i>, <b>, <u>,
# <ruby>, <rt>), classes (<c.class>) and inline timestamps (<00:00:01.000>). One pattern
# covers the whole tag, which is safe on text that only ever holds subtitle markup.
_INLINE_TAG = re.compile(r"</?v[^>]*>|</?[a-zA-Z][a-zA-Z0-9._-]*[^>]*>|<\d[^>]*>")
# A voice tag's speaker, from the text this connector has already stripped of other markup.
_VOICE_TAG = re.compile(r"\A<\s*v\s+(.+?)>")
# A leading dash pair marks a dialogue line in plain SRT ("- text" or "- text - text").
_DASHES = re.compile(r"\A\s*-\s+")


class SubtitlesSourceError(RuntimeError):
    """Base error. Messages carry folder-relative paths only, never absolute paths."""


class SubtitlesSourceUnavailableError(SubtitlesSourceError):
    """The folder or file is gone at run time, or the folder is empty though files were synced."""


class SubtitlesParseError(SubtitlesSourceError):
    """A file is not a valid SRT or WebVTT transcript; the run stops before anything is forgotten."""


@dataclass(frozen=True)
class _Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def _seconds(hours: str | None, minutes: str, seconds: str, millis: str) -> float:
    return (
        (int(hours) * 3600 if hours else 0)
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000
    )


def _strip_tags(text: str) -> tuple[str | None, str]:
    """The cue's speaker and its plain text: markup removed, whitespace collapsed.

    A leading voice tag (<v Name>) names the speaker; the tag is removed from the text.
    """
    speaker = None
    voice = _VOICE_TAG.match(text)
    if voice:
        # The spec allows an annotation after the name: <v Fred (upset)>; the parenthetical
        # is not part of who spoke.
        speaker = re.sub(r"\s*\([^)]*\)\s*\Z", "", voice.group(1)).strip()
        text = text[voice.end() :]
    text = _INLINE_TAG.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return speaker, text


def _parse_timestamp_block(block: str) -> list[_Cue]:
    """The cues of one block: a timing line, then one or more lines of text."""
    lines = [line for line in block.split("\n") if line.strip()]
    if len(lines) < 2:
        raise SubtitlesParseError(f"cue block has no timing line: {lines[0][:80] if lines else ''}")
    # The timing line comes first in a bare block, after the cue identifier line in a
    # WebVTT cue or the sequence number in an SRT one. Either way it is the line with
    # the arrow, and the text starts on the next line.
    timing_index = 0 if "-->" in lines[0] else 1
    timing = lines[timing_index]
    if "-->" not in timing:
        raise SubtitlesParseError(f"no timing line in cue block: {lines[0][:80]}")
    match = _TIMING.match(timing.strip())
    if match is None:
        raise SubtitlesParseError(f"unparseable timing line: {timing[:80]}")
    start = _seconds(*match.groups()[:4])
    end = _seconds(*match.groups()[4:])
    if end < start:
        raise SubtitlesParseError(f"cue ends before it starts: {timing[:80]}")
    speaker, text = _strip_tags(" ".join(lines[timing_index + 1 :]))
    return [_Cue(start, end, speaker, text)]


def _parse_blocks(text: str) -> list[_Cue]:
    """Every cue of a transcript, in file order. Blank lines separate the blocks."""
    cues: list[_Cue] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if _SKIP_BLOCK.match(block):
            continue
        # WebVTT cues may carry a cue identifier line before the timing line; SRT carries
        # a sequence number. Both are ignored, whatever they say.
        cues.extend(_parse_timestamp_block(block))
    return cues


def _parse_transcript(text: str, suffix: str) -> list[_Cue]:
    """The cues of one file. A WebVTT header is skipped; anything else is not accepted.

    A file that is not a transcript at all (no cue anywhere) is malformed: an empty export
    would otherwise look like a deleted file's content and re-cognify as an empty document.
    """
    if suffix == ".vtt":
        text = re.sub(r"\AWEBVTT[^\n]*\n?", "", text)
    cues = _parse_blocks(text)
    if not cues:
        raise SubtitlesParseError("no cues in file")
    return cues


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _paragraphs(cues: list[_Cue], min_gap: float) -> list[str]:
    """Consecutive cues merged into readable paragraphs.

    A cue joins the open paragraph when the speaker did not change (both None counts as the
    same, unnamed speaker) and no more than ``min_gap`` seconds of silence precede it. Every
    paragraph is rendered as ``[HH:MM:SS] Speaker: text``, without the speaker part when no
    voice tag named one.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    current_speaker: str | None = None
    current_start = 0.0
    last_end = 0.0

    def flush() -> None:
        if current:
            body = _DASHES.sub("", " ".join(current))
            who = f"{current_speaker}: " if current_speaker else ""
            paragraphs.append(f"[{_clock(current_start)}] {who}{body}".strip())

    for cue in cues:
        joins = current and cue.speaker == current_speaker and cue.start - last_end <= min_gap
        if not joins:
            flush()
            current, current_speaker, current_start = [], cue.speaker, cue.start
        current.append(cue.text)
        last_end = cue.end
    flush()
    return paragraphs


def _document_text(rel_path: str, cues: list[_Cue], min_gap: float) -> tuple[str, str]:
    """The document's title and its markdown body: one paragraph per merged cue run."""
    title = PurePosixPath(rel_path).stem
    body = "\n\n".join(_paragraphs(cues, min_gap))
    return title, body


def _file_id(rel_path: str, text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"subtitles:{rel_path}:{digest}"


def _ancestors(rel_path: str) -> list[dict]:
    folder = PurePosixPath(rel_path).parent
    return [
        {"kind": "folder", "id": parent.as_posix(), "name": parent.name}
        for parent in [folder, *folder.parents]
        if parent.parts
    ]


def _list_files(root: Path) -> list[str]:
    """Root-relative posix paths of the ``*.srt`` / ``*.vtt`` files to ingest, sorted.

    The same listing rules as the Obsidian source: regular files only, nothing hidden, no
    symlinks followed. A folder that cannot be listed raises, so an unreadable root never
    looks like a deleted set of files.
    """

    def refuse(error: OSError) -> None:
        rel = os.path.relpath(error.filename, root) if error.filename else error.strerror
        raise SubtitlesSourceUnavailableError(
            f"Could not list the folder {rel!r} ({type(error).__name__}); "
            "nothing was synced and nothing was forgotten."
        ) from None

    files = []
    for directory, subdirs, names in os.walk(root, onerror=refuse):
        subdirs[:] = [name for name in subdirs if not name.startswith(".")]
        for name in names:
            path = os.path.join(directory, name)
            if (
                name.lower().endswith(SUBTITLE_SUFFIXES)
                and not name.startswith(".")
                and not os.path.islink(path)
                and os.path.isfile(path)
            ):
                files.append(Path(path).relative_to(root).as_posix())
    return sorted(files)


def _read(text: bytes) -> str:
    """The transcript text. UTF-8 with or without BOM; UTF-16 exports are refused loudly."""
    try:
        return text.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise SubtitlesParseError(
            "not a UTF-8 transcript (a UTF-16 export must be converted first)"
        ) from error


def _iter_rows(
    root: Path,
    state: dict,
    stats: dict[str, int],
    min_gap: float,
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a document row for every transcript, then a tombstone for every gone one.

    Every file is parsed before the first row is yielded, so a malformed file stops the run
    with nothing published and nothing forgotten: a partial snapshot must not look like a
    deletion. A file removed from the folder since the last complete run is forgotten; the
    state is written after the last row, so a run that raises keeps the old one.
    """
    if not root.is_dir():
        raise SubtitlesSourceUnavailableError(
            "The transcript folder or file is not available; nothing was synced and nothing "
            "was forgotten. Check that the drive is mounted or point source_path at the new "
            "location."
        )
    listed = _list_files(root)
    previous = set(state.get("members", []))
    if previous and not os.listdir(root):
        raise SubtitlesSourceUnavailableError(
            "The transcript folder is empty, but earlier runs synced files from it; nothing "
            "was synced and nothing was forgotten. A drive that did not mount looks exactly "
            "like this. Check the mount, or leave one file in the folder to forget everything "
            "on purpose."
        )

    rows: list[dict[str, Any]] = []
    present: set[str] = set()
    for rel_path in listed:
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a failed file
        text = _read((root / rel_path).read_bytes())
        cues = _parse_transcript(text, PurePosixPath(rel_path).suffix.lower())
        title, body = _document_text(rel_path, cues, min_gap)
        file_id = _file_id(rel_path, text)
        present.add(file_id)
        row: dict[str, Any] = {
            "id": file_id,
            "title": title,
            "content": body,
            "url": "",
            "_deleted": False,
        }
        ancestors = _ancestors(rel_path)
        if ancestors:
            row[STRUCTURE_COLUMN] = {"ancestors": ancestors}
        rows.append(row)

    for row in rows:
        yield row
        # After the yield: a row that check_active stopped before it was published is not counted.
        stats["synced"] += 1
    for file_id in sorted(previous - present):
        yield {"id": file_id, "_deleted": True}
        stats["deleted"] += 1
    state.update(v=1, members=sorted(present))
    logger.info(
        "Subtitles: %d file(s) synced, %d forgotten.",
        stats["synced"],
        stats["deleted"],
    )


def subtitles_source(
    *,
    source_path: str | os.PathLike[str],
    resource_name: str | None = None,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per subtitle file of a folder or file.

    Hand the result to ``cognee.remember(...)`` with ``write_disposition="merge"`` and
    ``primary_key="id"``. A run walks the whole folder; a file whose text did not change makes
    the same row id, so nothing is re-cognified. A file removed from the folder is forgotten
    on the next run.

    Args:
        source_path: A folder walked recursively, or one ``.srt`` / ``.vtt`` file. A
            ``ValueError`` if it does not exist or is neither. A run raises if it has
            disappeared by then.
        resource_name: Names the staging table and the sync state. The default is derived
            from the resolved path, so moving the folder changes it and ingests everything
            again; pass a name to pin it. Two folders in one dataset need different names.
        min_gap_seconds: Silence between cues under which they merge into one paragraph;
            a speaker change always starts a paragraph. Must be >= 0.
        check_active: Host authorization checkpoint, called before each file is read and
            before each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Subtitles sync requires a Cognee build that reads per-row node sets and "
            "structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
        )
    if not isinstance(min_gap_seconds, (int, float)) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds must be a number >= 0.")
    root = Path(source_path).expanduser().resolve()
    if not root.exists():
        raise ValueError("source_path must be an existing folder or subtitle file.")
    if root.is_file():
        if root.suffix.lower() not in SUBTITLE_SUFFIXES:
            raise ValueError(
                f"source_path must be a .srt or .vtt file or a folder, not {root.suffix!r}."
            )
        root = root.parent
    elif not root.is_dir():
        raise ValueError("source_path must be an existing folder or subtitle file.")
    if resource_name is None:
        resource_name = f"subtitles_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")
    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def subtitle_files():
        stats.update(synced=0, deleted=0)
        rows = _iter_rows(root, dlt.current.resource_state(), stats, min_gap_seconds, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = subtitle_files()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "subtitles")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
