"""Bounded LanceDB compaction, shared by the in-process adapter and the
subprocess worker. Imports ``lance`` (pylance) lazily. Must not import cognee.

Why not ``table.optimize()``: LanceDB's own ``optimize`` merges every fragment
below its default target of about 1M rows. Every cognee table is smaller than
that, so each call rewrites the whole table -- seconds on an SSD, minutes on a
laptop disk or a cloud volume, and at cognee's write rate hundreds of GB of
disk writes a day. Lance's lower-level compaction API takes a fragment target.
With a 20k-row target (~250 MB at 3072 dims) only the small, recently written
fragments are merged and the large cold ones are left alone.

A pass has two halves, run separately because only the first one conflicts
with writers:

* ``compact_fragments`` rewrites small fragments into larger ones. Its commit
  is a Lance "rewrite" transaction, so callers serialise it with their own
  writers.
* ``prune_superseded_versions`` deletes old manifests and the files only they
  reference. It commits nothing, so it can run beside writers: the files of an
  in-progress write are referenced by no manifest yet, and Lance keeps such
  "unverified" files for 7 days (``delete_unverified`` is never set).

Both halves take a count bound -- ``max_tasks`` compaction tasks,
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

Requires ``pylance`` built on the same Lance core ``lancedb`` bundles (pyproject
keeps the two on matching release lines); a mismatched pylance cannot read the
files lancedb wrote, which the adapter detects and turns compaction off for.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Any

DEFAULT_TARGET_ROWS_PER_FRAGMENT = 20_000
DEFAULT_RETENTION_SECONDS = 300
DEFAULT_MAX_TASKS_PER_RUN = 4
DEFAULT_MAX_VERSIONS_PER_RUN = 1_000
#: Fragments with at least this share of deleted rows are rewritten even when
#: already at target size, so the dead rows that re-upserts leave in cold data
#: are reclaimed once they reach a fifth of a fragment.
MATERIALIZE_DELETIONS_THRESHOLD = 0.2


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
    now = datetime.now(versions[-1]["timestamp"].tzinfo)
    window = max(0, int(retention_seconds))
    removable = []
    for version, successor in pairwise(versions):
        if (now - successor["timestamp"]).total_seconds() < window:
            break
        removable.append(int(version["version"]))
    return removable


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
