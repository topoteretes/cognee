"""Letterboxd connector for cognee: a ``dlt`` source that turns a Letterboxd
data export (CSV) into memory.

A Letterboxd member exports their data from Settings → Import & Export →
Export as a ZIP (or an unzipped folder) containing ``watched.csv``,
``ratings.csv``, ``reviews.csv`` and ``diary.csv``, plus one CSV per curated
list. This source reads the export, merges every file into one document per
film keyed by the film's Letterboxd URI (name, year, rating, review, diary
date, rewatch notes), and one document per list (name + film roster).

Hand the result to ``cognee.remember(...)`` with
``write_disposition="merge"`` and ``primary_key="id"``::

    import cognee
    from cognee.tasks.ingestion.connectors import letterboxd_source

    await cognee.remember(
        letterboxd_source(export_path="/path/to/letterboxd-export.zip"),
        dataset_name="my_letterboxd",
        primary_key="id",
        write_disposition="merge",
    )

Design
------
* **Auth**: none. The user downloads the export themselves and passes the
  folder or ZIP path.
* **Transport**: the standard library only (``csv``, ``zipfile``,
  ``pathlib``), so this connector adds no dependency beyond cognee's core.
* **One row per film**: the Letterboxd film URI is the stable primary key.
  Rows from ``watched.csv``, ``ratings.csv``, ``reviews.csv`` and ``diary.csv``
  are merged into a single document per film; a diary rewatch entry is folded
  into the film's document. A film that appears in several files keeps one row.
* **One row per list**: every ``lists/*.csv`` (or a top-level ``lists.csv``)
  becomes one document whose content is the list name plus its film roster.
* **Incremental**: a run re-renders every film and list from the export and
  only emits a row when its rendered content changed since the last run
  (deterministic rendering, so a re-export with no edits re-emits nothing).
* **Forget-on-delete**: a film present in the previous export but missing from
  the new one is emitted as a ``_deleted`` tombstone, so the next sync forgets
  it. A transient read error aborts before any tombstone, so a partial read
  never looks like a mass deletion.
* **Node set**: every row is tagged ``letterboxd:<source_id>`` where
  ``source_id`` is derived from the resolved export path, so a Letterboxd
  source is scoped to its export.

Limitations
-----------
* Letterboxd has no public API; the CSV export is the only access path, so a
  sync always starts from a fresh manual export.
* The export's exact column headers can vary by export date; unknown columns
  are ignored and rows lacking a film name or URI are skipped.
* List CSVs are read from a ``lists/`` folder or a top-level ``lists.csv``;
  other list layouts are not recognized.
"""

import csv
import hashlib
import io
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from cognee.shared.logging_utils import get_logger
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)

logger = get_logger("letterboxd_connector")

_SOURCE_PREFIX = "letterboxd:"

# Column aliases Letterboxd's export may use, checked in order. Headers are
# matched case-insensitively after stripping surrounding whitespace.
_FILM_ALIASES = ("film", "film name", "title", "name")
_YEAR_ALIASES = ("year", "released year", "release year")
_URI_ALIASES = ("letterboxd uri", "letterboxd_url", "uri", "url", "link")
_RATING_ALIASES = ("rating", "my rating", "score")
_REVIEW_ALIASES = ("review", "my review", "body", "text")
_DATE_ALIASES = ("watched date", "date", "watched", "diary date")
_REWATCH_ALIASES = ("rewatch", "is rewatch", "rewatch?")
_LIST_NAME_ALIASES = ("list name", "list", "name")


class LetterboxdSourceError(RuntimeError):
    """Base error for the Letterboxd connector."""


class LetterboxdExportUnavailableError(LetterboxdSourceError):
    """The export folder or ZIP is gone at run time."""


def _source_id(export_path: Path) -> str:
    """Stable id for an export, derived from its resolved path."""
    return _SOURCE_PREFIX + hashlib.sha256(str(export_path).encode()).hexdigest()[:16]


def _first(values: dict[str, Any], aliases: tuple[str, ...]) -> str:
    """Return the first non-empty value for any of ``aliases`` (case-insensitive)."""
    lower = {str(key).strip().lower(): value for key, value in values.items()}
    for alias in aliases:
        value = lower.get(alias)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _row_uri(row: dict[str, Any]) -> str:
    return _first(row, _URI_ALIASES)


def _film_row(
    uri: str,
    title: str,
    year: str,
    rating: str,
    review: str,
    date: str,
    rewatch: str,
    source_id: str,
) -> dict[str, Any]:
    """One document row for a film."""
    lines = [
        f"Title: {title}" if title else "",
        f"Year: {year}" if year else "",
        f"URI: {uri}",
        f"Rating: {rating}" if rating else "",
        f"Review: {review}" if review else "",
        f"Date: {date}" if date else "",
        f"Rewatch: {rewatch}" if rewatch else "",
    ]
    content = "\n".join(line for line in lines if line)
    return {
        "id": uri,
        "title": title or uri,
        "content": content,
        "url": uri,
        "_deleted": False,
        NODE_SET_COLUMN: [source_id],
        STRUCTURE_COLUMN: {
            "ancestors": [{"kind": "export", "id": source_id, "name": "Letterboxd export"}]
        },
    }


def _list_row(list_name: str, films: list[dict[str, str]], source_id: str) -> dict[str, Any]:
    """One document row for a curated list."""
    lines = [f"List: {list_name}" if list_name else "List", "Films:"]
    for film in films:
        parts = []
        if film.get("title"):
            parts.append(film["title"])
        if film.get("year"):
            parts.append(f"({film['year']})")
        if film.get("uri"):
            parts.append(film["uri"])
        lines.append("- " + " ".join(parts))
    content = "\n".join(lines)
    list_id = (
        "list:" + hashlib.sha256(list_name.encode()).hexdigest()[:16]
        if list_name
        else hashlib.sha256(str(films).encode()).hexdigest()[:16]
    )
    return {
        "id": list_id,
        "title": list_name or "Untitled list",
        "content": content,
        "url": "",
        "_deleted": False,
        NODE_SET_COLUMN: [source_id],
        STRUCTURE_COLUMN: {
            "ancestors": [{"kind": "export", "id": source_id, "name": "Letterboxd export"}]
        },
    }


def _is_list_file(rel_path: str) -> bool:
    """True when a CSV row belongs to a curated-list file."""
    parts = [part.lower() for part in Path(rel_path).parts]
    name = Path(rel_path).name.lower()
    return "lists" in parts or name == "lists.csv" or name.startswith("lists")


def _read_export_rows(export_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(relative_path, row_dict)`` for every data row in a Letterboxd export.

    ``export_path`` may be a folder or a ZIP. Only CSV files are read; rows
    lacking both a film name and a URI are skipped by the caller.
    """
    if not export_path.exists():
        raise LetterboxdExportUnavailableError(
            f"Letterboxd export {export_path!r} does not exist; nothing was "
            "synced and nothing was forgotten. Point export_path at the "
            "unzipped export folder or the exported ZIP."
        )
    if export_path.is_dir():
        files = sorted(p for p in export_path.rglob("*.csv") if p.is_file())
        for path in files:
            rel_path = path.relative_to(export_path).as_posix()
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield from _iter_csv(handle, rel_path)
    elif zipfile.is_zipfile(export_path):
        with zipfile.ZipFile(export_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".csv"):
                    continue
                with archive.open(name) as handle:
                    text = io.TextIOWrapper(handle, encoding="utf-8-sig")
                    yield from _iter_csv(text, name)
    else:
        raise LetterboxdExportUnavailableError(
            f"Letterboxd export {export_path!r} is not a folder or a ZIP; "
            "nothing was synced and nothing was forgotten."
        )


def _iter_csv(handle, rel_path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        return
    for row in reader:
        yield rel_path, row


def _iter_rows(
    export_path: Path,
    state: dict,
    stats: dict[str, int],
    check_active: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a row per changed/new film and list, then deletion tombstones.

    ``state`` holds ``{"films": {uri: row}, "lists": {list_id: row}}`` from the
    prior run. A film or list whose rendered content is unchanged is not
    re-emitted. A transient read error propagates before ``state`` is touched,
    so a partial read never produces tombstones. A film present before but
    absent now is tombstoned.
    """
    source_id = _source_id(export_path)

    stats.clear()
    stats.update(
        films_scanned=0,
        films_changed=0,
        films_unchanged=0,
        lists_scanned=0,
        lists_changed=0,
        deleted=0,
    )

    previous_films = state.get("films", {})
    previous_lists = state.get("lists", {})

    present_films: dict[str, dict[str, Any]] = {}
    present_lists: dict[str, dict[str, Any]] = {}
    films_by_uri: dict[str, dict[str, str]] = {}
    roster_by_list: dict[str, list[dict[str, str]]] = {}

    for rel_path, row in _read_export_rows(export_path):
        if check_active is not None:
            check_active()  # outside any try: a revoked connection is not a bad row
        uri = _row_uri(row)
        title = _first(row, _FILM_ALIASES)
        year = _first(row, _YEAR_ALIASES)
        rating = _first(row, _RATING_ALIASES)
        review = _first(row, _REVIEW_ALIASES)
        date = _first(row, _DATE_ALIASES)
        rewatch = _first(row, _REWATCH_ALIASES)

        # A row that names a film is a film record, no matter which file it
        # came from; merge every file's take into one record per URI, with the
        # diary/review text winning on conflict. Rows that name neither a film
        # nor a URI cannot be keyed and are skipped.
        if title or uri:
            record = films_by_uri.setdefault(
                uri or "",
                {"title": title, "year": year, "rating": rating,
                 "review": review, "date": date, "rewatch": rewatch},
            )
            if review:
                record["review"] = review
            if rating:
                record["rating"] = rating
            if date:
                record["date"] = date
            if rewatch:
                record["rewatch"] = rewatch
            stats["films_scanned"] += 1

        # A curated-list file contributes its rows to that list's roster.
        if _is_list_file(rel_path):
            list_name = _first(row, _LIST_NAME_ALIASES) or Path(rel_path).stem
            film_uri = _row_uri(row)
            film_title = _first(row, _FILM_ALIASES)
            film_year = _first(row, _YEAR_ALIASES)
            roster_by_list.setdefault(list_name, [])
            if film_uri or film_title:
                roster_by_list[list_name].append(
                    {"title": film_title, "year": film_year, "uri": film_uri}
                )
            stats["lists_scanned"] += 1

    for uri, record in films_by_uri.items():
        if not uri:
            continue
        row = _film_row(
            uri,
            record["title"],
            record["year"],
            record["rating"],
            record["review"],
            record["date"],
            record["rewatch"],
            source_id,
        )
        previous = {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "url": row["url"],
        }
        if previous_films.get(uri) != previous:
            stats["films_changed"] += 1
            yield row
        else:
            stats["films_unchanged"] += 1
        present_films[uri] = previous

    for list_name, roster in roster_by_list.items():
        row = _list_row(list_name, roster, source_id)
        previous = {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "url": row["url"],
        }
        list_id = row["id"]
        if previous_lists.get(list_id) != previous:
            stats["lists_changed"] += 1
            yield row
        present_lists[list_id] = previous

    deleted_uris = sorted(set(previous_films) - set(present_films))
    deleted_list_ids = sorted(set(previous_lists) - set(present_lists))
    for uri in deleted_uris:
        stats["deleted"] += 1
        yield {"id": uri, "_deleted": True}
    for list_id in deleted_list_ids:
        stats["deleted"] += 1
        yield {"id": list_id, "_deleted": True}

    state.update(
        films=present_films,
        lists=present_lists,
    )
    logger.info(
        "Letterboxd: %d film(s) scanned (%d changed, %d unchanged), "
        "%d list(s) (%d changed), %d deletion(s).",
        stats["films_scanned"],
        stats["films_changed"],
        stats["films_unchanged"],
        stats["lists_scanned"],
        stats["lists_changed"],
        stats["deleted"],
    )


def letterboxd_source(
    export_path: str | Path,
    resource_name: str | None = None,
    check_active: Callable[[], None] | None = None,
):
    """Return a ``dlt`` resource that yields one document per film and list of a
    Letterboxd export.

    Hand the result to ``cognee.remember(...)`` with
    ``write_disposition="merge"`` and ``primary_key="id"``. A run re-renders the
    whole export; a film or list whose content did not change costs no
    extraction. A normal return can still have skipped rows: read
    ``cognee_sync_stats["deleted"]`` afterwards.

    Args:
        export_path: The unzipped export folder or the exported ZIP. A
            ``ValueError`` if it does not exist. A run raises if it has
            disappeared by then.
        resource_name: Names the staging table and the sync state. The default
            is derived from the resolved export path, so moving the export
            changes it and re-ingests every film again; pass a name to pin it.
        check_active: Host authorization checkpoint called before each row is
            read and before each row is published.
    """
    import dlt

    if getattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 0) < 3:
        raise RuntimeError(
            "Letterboxd sync requires a Cognee build that reads per-row node "
            "sets and structure (DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee "
            "before syncing."
        )

    resolved = Path(export_path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError("export_path must point to an existing folder or ZIP.")

    source_id = _source_id(resolved)
    if resource_name is None:
        resource_name = f"letterboxd_{source_id.split(':')[1]}"
    elif not isinstance(resource_name, str) or not resource_name.strip():
        raise ValueError("resource_name must be a non-empty string.")

    stats: dict[str, int] = {}

    @dlt.resource(
        name=resource_name,
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def letterboxd_documents():
        stats.update(
            films_scanned=0,
            films_changed=0,
            films_unchanged=0,
            lists_scanned=0,
            lists_changed=0,
            deleted=0,
        )
        rows = _iter_rows(resolved, dlt.current.resource_state(), stats, check_active)
        yield from dlt_utils.guarded_rows(rows, check_active)

    resource = letterboxd_documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "letterboxd")
    setattr(resource, PIPELINE_SCOPE_ATTR, resource_name)
    resource.cognee_sync_stats = stats
    return resource
