"""Bounded LanceDB compaction, shared by the in-process adapter and the
subprocess worker. Imports ``lance`` (pylance) lazily. Must not import cognee.

Why not ``table.optimize()``: LanceDB's own ``optimize`` merges every fragment
below its default target of about 1M rows. Every cognee table is smaller than
that, so each call rewrites the whole table -- seconds on an SSD, minutes on a
laptop disk or a cloud volume, and at cognee's write rate hundreds of GB of
disk writes a day. Lance's lower-level compaction API takes a fragment target.
With a 20k-row target (~250 MB at 3072 dims) only the small, recently written
fragments are merged and the large cold ones are left alone, so the cost of a
run is bounded by one warm fragment regardless of table size.

A plan can also be executed partially. A store that already carries a large
backlog (tens of thousands of fragments, see issue #4684) is drained
``max_tasks`` tasks per run instead of blocking one pipeline run for an hour on
a slow disk.

Superseded versions are pruned behind a retention window, and the window is
measured from when a version was SUPERSEDED, not from when it was written.
Lance's own ``cleanup_old_versions`` ages a version by its commit time, which
is unsafe on an idle table: an hour-old version that is still the latest can be
opened by a reader and then deleted a second later, right after the compaction
that supersedes it. ``_cleanup_superseded_versions`` removes a version only
once its successor is at least ``retention_seconds`` old, so any reader whose
single read is shorter than the window is safe, in this or any other process
(only manifest timestamps are consulted). The files are therefore reclaimed by
the pass that runs after the window, not by the pass that compacted.
``delete_unverified`` is never set -- the files of an in-progress write from
another process look exactly like the leftovers of a failed one.

Requires ``pylance`` built on the same Lance core ``lancedb`` bundles (the two
are pinned as a pair in pyproject); a mismatched pylance cannot read the files
lancedb wrote.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

DEFAULT_TARGET_ROWS_PER_FRAGMENT = 20_000
DEFAULT_RETENTION_SECONDS = 300
DEFAULT_MAX_TASKS_PER_RUN = 4
#: Fragments with at least this share of deleted rows are rewritten even when
#: already at target size, so the dead rows that re-upserts leave in cold data
#: are reclaimed once they reach a fifth of a fragment.
MATERIALIZE_DELETIONS_THRESHOLD = 0.2


def compact_dataset(
    dataset,
    *,
    target_rows_per_fragment: int = DEFAULT_TARGET_ROWS_PER_FRAGMENT,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    max_tasks: int = DEFAULT_MAX_TASKS_PER_RUN,
) -> dict[str, Any]:
    """Compact ``dataset`` (a ``lance.LanceDataset``) and prune versions older
    than the retention window.

    Synchronous and I/O-bound: callers run it in a worker thread. Executes at
    most ``max_tasks`` of the planned compaction tasks (``0`` = all of them,
    negative = none: plan and prune only, used when a shared budget is spent)
    and commits only those, which Lance supports explicitly. Returns plain
    ints so the result crosses the subprocess boundary without pickling
    anything but builtins.
    """
    import lance
    from lance.optimize import Compaction, CompactionOptions

    options = CompactionOptions(
        target_rows_per_fragment=int(target_rows_per_fragment),
        materialize_deletions=True,
        materialize_deletions_threshold=MATERIALIZE_DELETIONS_THRESHOLD,
    )
    plan = Compaction.plan(dataset, options)
    tasks = list(plan.tasks)
    if max_tasks < 0:
        tasks = []
    elif max_tasks > 0:
        tasks = tasks[:max_tasks]

    stats: dict[str, Any] = {
        "planned_tasks": plan.num_tasks(),
        "executed_tasks": len(tasks),
        "fragments_removed": 0,
        "fragments_added": 0,
        "old_versions_removed": 0,
        "bytes_removed": 0,
    }
    if tasks:
        metrics = Compaction.commit(dataset, [task.execute(dataset) for task in tasks])
        stats["fragments_removed"] = int(metrics.fragments_removed)
        stats["fragments_added"] = int(metrics.fragments_added)

    # Re-open so the cleanup sees the version committed just above.
    cleanup = _cleanup_superseded_versions(lance.dataset(dataset.uri), int(retention_seconds))
    if cleanup is not None:
        stats["old_versions_removed"] = int(cleanup.old_versions)
        stats["bytes_removed"] = int(cleanup.bytes_removed)
    return stats


def _cleanup_superseded_versions(dataset, retention_seconds: int):
    """Remove every version whose successor is at least ``retention_seconds`` old.

    ``cleanup_old_versions`` only takes an age cut-off measured against each
    version's own commit time, so the cut-off is derived from the version list:
    find the newest version that is itself older than the window (call it K);
    every version older than K has a successor no newer than K, hence one that
    has aged past the window, so all of them can go. K stays: its successor may
    have been committed a moment ago, and a reader may have opened K just
    before that. The cut-off sits 1 ms before K: Lance evaluates it against
    its own clock a little after ours, and that margin keeps K out of reach. A
    version committed within that millisecond before K (a compaction commits a
    twin of the previous state right before the rewrite) is kept until a pass
    in which a newer version has aged, i.e. the pass after the next write.
    Returns Lance's cleanup stats, or ``None`` when nothing is removable yet.
    Lance always keeps the latest version regardless.
    """
    versions = dataset.versions()  # oldest -> newest
    if len(versions) < 2:
        return None
    stamps = [version["timestamp"] for version in versions]
    now = datetime.now(stamps[-1].tzinfo)
    window = max(0, retention_seconds)
    aged = [i for i, stamp in enumerate(stamps) if (now - stamp).total_seconds() >= window]
    if not aged or aged[-1] == 0:
        return None
    keep_from = stamps[aged[-1]]
    return dataset.cleanup_old_versions(older_than=(now - keep_from) + timedelta(milliseconds=1))
