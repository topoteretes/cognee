"""Bounded LanceDB compaction, shared by the in-process adapter and the
subprocess worker. Imports ``lance`` (pylance) lazily. Must not import cognee.

Why not ``table.optimize()``: LanceDB's own ``optimize`` merges every fragment
below its default target of about 1M rows. Every cognee table is smaller than
that, so each call rewrites the whole table -- seconds on an SSD, minutes on a
laptop disk or a cloud volume, and at cognee's write rate hundreds of GB of
disk writes a day. Lance's lower-level compaction API takes a fragment target.
With a 20k-row target (~250 MB at 3072 dims) only the small, recently written
fragments are merged and the large cold ones are left alone.

A pass has three parts, run separately because only the first one conflicts
with writers:

* ``compact_fragments`` rewrites small fragments into larger ones. Its commit
  is a Lance "rewrite" transaction, so callers serialise it with their own
  writers.
* ``prune_superseded_versions`` deletes old manifests and the files only they
  reference. It commits nothing, so it can run beside writers: the files of an
  in-progress write are referenced by no manifest yet, and Lance keeps such
  "unverified" files for 7 days (``delete_unverified`` is never set).
* ``refresh_inverted_indices`` folds the rows written since the last pass into
  the table's full-text (BM25) indexes, which Lance never does on its own; a
  query scans the unindexed rows until then. Bounded by what the run wrote,
  plus one merge of an index's delta segments every ``max_segments`` passes.

The first two take a count bound -- ``max_tasks`` compaction tasks,
``max_versions`` deleted versions -- and leave the rest for the next pass. A
store that already carries a large backlog (tens of thousands of fragments and
versions, see issue #4684) is drained over several cognify runs instead of
stalling one of them for half an hour.

Reader safety: a version is deleted only once its SUCCESSOR is at least
``retention_seconds`` old. A reader holding version V keeps working as long as
its read is shorter than the window, in this or any other process. The
versions to delete are passed to Lance by number, so no clock comparison
happens inside Lance and a scheduling delay between our clock read and Lance's
cannot move the cut-off onto a version a reader just opened.

Requires ``pylance`` built on the same Lance core ``lancedb`` bundles. A pylance
on an older core may be unable to read what lancedb wrote; one on a newer core
reads it fine but may commit a version lancedb's core cannot read. lancedb
exposes its bundled core only inside its native binary, so
``LANCE_CORE_BY_LANCEDB`` records it per lancedb release line, and
``lance_core_mismatch`` turns compaction off when the installed pair is not a
known match (pyproject keeps the two on matching lines; this catches an
override).
"""

from __future__ import annotations

import time
from datetime import datetime
from itertools import pairwise
from typing import Any

DEFAULT_TARGET_ROWS_PER_FRAGMENT = 20_000
DEFAULT_RETENTION_SECONDS = 300
DEFAULT_MAX_TASKS_PER_RUN = 4
DEFAULT_MAX_VERSIONS_PER_RUN = 1_000
#: Delta segments a full-text index may accumulate before a pass merges them
#: into one (``refresh_inverted_indices``).
DEFAULT_MAX_INDEX_SEGMENTS = 8
#: Fragments with at least this share of deleted rows are rewritten even when
#: already at target size, so the dead rows that re-upserts leave in cold data
#: are reclaimed once they reach a fifth of a fragment.
MATERIALIZE_DELETIONS_THRESHOLD = 0.2


#: Lance core release line (``major.minor``) bundled by each lancedb release line
#: (``strings _lancedb*.so | grep lance-core`` shows it). Update together with
#: the lancedb/pylance ranges in pyproject.
LANCE_CORE_BY_LANCEDB = {"0.39": "12.0"}


def _release_line(version: str) -> str:
    return ".".join(version.split(".")[:2])


def lance_core_mismatch() -> str | None:
    """Why the installed lancedb and pylance must not be used together, or ``None``."""
    import lance
    import lancedb

    lancedb_version = getattr(lancedb, "__version__", "unknown")
    pylance_version = getattr(lance, "__version__", "unknown")
    expected = LANCE_CORE_BY_LANCEDB.get(_release_line(lancedb_version))
    if expected is None:
        return (
            f"lancedb {lancedb_version} is not a release line with a known bundled "
            f"Lance core (known: {sorted(LANCE_CORE_BY_LANCEDB)})"
        )
    if _release_line(pylance_version) != expected:
        return (
            f"pylance {pylance_version} is not on Lance {expected}.x, the core "
            f"lancedb {lancedb_version} bundles"
        )
    return None


class PylanceIncompatibleError(RuntimeError):
    """pylance cannot open a table lancedb wrote: the two are built on
    different Lance cores. Not specific to one table, so callers stop
    compacting altogether instead of retrying every table."""


async def open_as_lance(table):
    """``table`` (a ``lancedb.AsyncTable``) as a pylance dataset at its latest version."""
    try:
        dataset = await table.to_lance()
        dataset.versions()
    except Exception as exc:
        raise PylanceIncompatibleError(f"{type(exc).__name__}: {exc}") from exc
    return dataset


def _bounded(items: list, limit: int) -> list:
    """``limit`` items: ``0`` = all of them, negative = none (a shared budget
    already spent elsewhere)."""
    if limit < 0:
        return []
    return items[:limit] if limit > 0 else items


def compact_fragments(
    dataset,
    *,
    target_rows_per_fragment: int = DEFAULT_TARGET_ROWS_PER_FRAGMENT,
    max_tasks: int = DEFAULT_MAX_TASKS_PER_RUN,
) -> dict[str, Any]:
    """Merge the small fragments of ``dataset`` (a ``lance.LanceDataset``).

    Synchronous and I/O-bound: callers run it in a worker thread. Executes at
    most ``max_tasks`` of the planned compaction tasks (``0`` = all of them,
    negative = none: plan only, used when a shared budget is spent) and
    commits only those, which Lance supports explicitly. Returns plain ints so
    the result crosses the subprocess boundary without pickling anything but
    builtins.
    """
    from lance.optimize import Compaction, CompactionOptions

    options = CompactionOptions(
        target_rows_per_fragment=int(target_rows_per_fragment),
        materialize_deletions=True,
        materialize_deletions_threshold=MATERIALIZE_DELETIONS_THRESHOLD,
    )
    plan = Compaction.plan(dataset, options)
    tasks = _bounded(list(plan.tasks), int(max_tasks))

    stats: dict[str, Any] = {
        "planned_tasks": plan.num_tasks(),
        "executed_tasks": len(tasks),
        "fragments_removed": 0,
        "fragments_added": 0,
    }
    if tasks:
        metrics = Compaction.commit(dataset, [task.execute(dataset) for task in tasks])
        stats["fragments_removed"] = int(metrics.fragments_removed)
        stats["fragments_added"] = int(metrics.fragments_added)
    return stats


def removable_versions(dataset, retention_seconds: int) -> list[int]:
    """Version numbers whose successor is at least ``retention_seconds`` old.

    ``versions()`` is ordered oldest to newest and commit timestamps only grow,
    so the result is always a prefix of the history, never the latest version.
    """
    versions = dataset.versions()
    if len(versions) < 2:
        return []
    now = time.time()
    window = max(0, int(retention_seconds))
    removable = []
    for version, successor in pairwise(versions):
        if now - _epoch_seconds(successor["timestamp"]) < window:
            break
        removable.append(int(version["version"]))
    return removable


def _epoch_seconds(stamp: datetime) -> float:
    """A ``versions()`` timestamp as epoch seconds, never older than it really is.

    pylance builds these as naive local time (``datetime.fromtimestamp``) and
    then adds the microseconds, which resets ``fold``. Subtracting naive
    datetimes is therefore off by the DST shift whenever one lies in between,
    and an hour repeated when clocks go back cannot be told apart. Epoch
    seconds fix the first; for the second, ``fold=1`` takes the later of the
    two candidate instants, so a version can only look younger than it is
    (kept up to an hour longer), never older (deleted while a reader may hold
    it). An aware timestamp, should pylance ever return one, is exact as is.
    """
    if stamp.tzinfo is not None:
        return stamp.timestamp()
    return stamp.replace(fold=1).timestamp()


def prune_superseded_versions(
    dataset,
    *,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    max_versions: int = DEFAULT_MAX_VERSIONS_PER_RUN,
) -> dict[str, Any]:
    """Delete the versions of ``dataset`` whose successor has aged past the window.

    Oldest first, at most ``max_versions`` of them (``0`` = all, negative =
    none); what is left is reported as ``versions_pending`` and goes in a later
    pass. Tagged versions are skipped, not raised on: cognee never tags, but a
    user's tag must not stop cleanup.
    """
    removable = removable_versions(dataset, retention_seconds)
    selected = _bounded(removable, int(max_versions))
    stats: dict[str, Any] = {
        "old_versions_removed": 0,
        "bytes_removed": 0,
        "versions_pending": len(removable) - len(selected),
    }
    if selected:
        cleanup = dataset.cleanup_old_versions(
            versions=selected, error_if_tagged_old_versions=False
        )
        stats["old_versions_removed"] = int(cleanup.old_versions)
        stats["bytes_removed"] = int(cleanup.bytes_removed)
    return stats


def refresh_inverted_indices(
    dataset, *, max_segments: int = DEFAULT_MAX_INDEX_SEGMENTS
) -> dict[str, Any]:
    """Fold the rows written since the last pass into the inverted (full-text) indexes.

    Lance never adds new rows to an existing index: a full-text query scans
    them, tokenizing every unindexed row on every query -- about 100x the
    indexed cost once the tail holds a few thousand chunks or a few hundred
    cached documents. ``optimize_indices`` with ``num_indices_to_merge=0``
    appends one delta segment holding exactly the unindexed rows (bounded by
    what the run wrote; a no-op that commits nothing when there are none).
    Query cost grows with the segment count, so once an index has
    ``max_segments`` of them the pass merges them all into one: the rewrite of
    the whole index is paid once per ``max_segments`` passes. Vector indexes
    (cognee builds none) and every other index type are left alone.

    Synchronous and I/O-bound like ``compact_fragments``: callers run it in a
    worker thread. Returns plain ints.
    """
    segments_by_index: dict[str, int] = {}
    for entry in dataset.list_indices():
        if entry.get("type") != "Inverted":
            continue
        name = entry["name"]
        segments_by_index[name] = segments_by_index.get(name, 0) + 1

    stats: dict[str, Any] = {
        "inverted_indices": len(segments_by_index),
        "inverted_indices_merged": 0,
    }
    merge_at = max(1, int(max_segments))
    for name, segments in segments_by_index.items():
        merge_all = segments >= merge_at
        dataset.optimize.optimize_indices(
            num_indices_to_merge=segments if merge_all else 0, index_names=[name]
        )
        if merge_all:
            stats["inverted_indices_merged"] += 1
    return stats
