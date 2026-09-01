"""Run manifests for eval capture (SDK-529).

A ``run_scope`` brackets one operation or pipeline run. It is a SYNC
``@contextmanager`` — the same shape as ``operation_usage_scope`` /
``parent_run_scope`` in ``cognee/modules/operations/usage_accumulator.py`` — so
it stacks on the existing ``with`` line inside the ``run_tasks`` async
generator. The ``RunScope`` is mutated IN PLACE and the contextvar is bound to
that one object before any task fan-out, so ``note()``/``bump()`` from
``create_task`` children reach the same accumulator. On exit a
``run.manifest`` event is enqueued (no drain here — callers ``await drain()``
themselves where appropriate). A caller whose scope outlives its drain —
run_tasks, whose ``with`` encloses the terminal ``yield`` — calls
``RunScope.finish()`` first so the drain covers the run's own manifest; the
exit path then skips the duplicate.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator
from uuid import UUID

from . import hook
from .events import KIND_RUN_MANIFEST


@dataclass
class RunScope:
    """Mutable per-run accumulator; the payload source of the run manifest."""

    run_id: UUID | str | None
    # Mutable — bound late via set_dataset() by record_operation's callers.
    dataset_id: UUID | str | None = None
    # "operation" | "pipeline"
    kind: str = "operation"
    # Decided once at scope entry; retrieval-kind events consult it via should_capture().
    sampled: bool = True
    started_at: float = field(default_factory=time.time)
    fields: dict[str, Any] = field(default_factory=dict)
    counters: Counter[str] = field(default_factory=Counter)
    dropped_at_enter: int = 0
    parent: RunScope | None = None
    # Set by finish(): the manifest has been enqueued, exit must not emit again.
    finished: bool = False

    def set_dataset(self, dataset_id: UUID | str | None) -> None:
        """Bind the dataset late. Because ``CaptureEvent.scope`` is resolved at
        serialization time, events already buffered for this run pick it up too;
        only events flushed BEFORE this call (≤ FLUSH_INTERVAL_S window) can land
        under ``nodataset/`` — the manifest is authoritative."""
        self.dataset_id = dataset_id

    def resolved_dataset_id(self) -> UUID | str | None:
        """This run's dataset, or the nearest enclosing run's when it has none.

        A ``record_operation``-wrapped search executed by a pipeline task opens an
        operation scope with no dataset of its own inside a pipeline scope that
        knows it; attributing to the parent keeps ``nodataset/`` for runs that
        are genuinely dataset-less.
        """
        scope: RunScope | None = self
        while scope is not None:
            if scope.dataset_id is not None:
                return scope.dataset_id
            scope = scope.parent
        return None

    def note(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def bump(self, counter: str, n: int = 1) -> None:
        self.counters[counter] += n

    def to_manifest(self) -> dict[str, Any]:
        ended_at = time.time()
        dataset_id = self.resolved_dataset_id()
        parent = self.parent
        parent_run_id = None if parent is None or parent.run_id is None else str(parent.run_id)
        return {
            # Fields first so the envelope keys below always win: a note() under
            # "run_id" or "kind" cannot re-file the manifest, and a note()
            # under "counters" is not silently discarded.
            **self.fields,
            "run_id": None if self.run_id is None else str(self.run_id),
            # Lets a nested run (an operation recorded inside a pipeline task)
            # be joined to its enclosing run offline.
            "parent_run_id": parent_run_id,
            "dataset_id": None if dataset_id is None else str(dataset_id),
            "kind": self.kind,
            "sampled": self.sampled,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_s": ended_at - self.started_at,
            "counters": dict(self.counters),
            # Process-global counter delta: an approximation under concurrent runs.
            "dropped_events": max(0, hook._dropped - self.dropped_at_enter),
        }

    def finish(self) -> None:
        """Enqueue the manifest now; idempotent.

        ``run_scope`` calls this on exit. A caller that drains BEFORE its scope
        closes (run_tasks: the scope encloses the terminal ``yield``) calls it
        right before the drain so the run's most important record is covered
        by that drain rather than left to the interval tick or the atexit hook.
        Manifests get headroom past the queue bound (up to ``2 * QUEUE_SIZE``
        buffered events in total): an overflowing run must still report
        ``dropped_events``, but the buffer stays finite under a wedged sink.

        An unsampled run still gets its manifest. Sampling gates ``retrieval.*``
        payloads (see ``should_capture``), not the record that identifies the run:
        every other kind is captured at full rate regardless, so suppressing the
        manifest would leave those events — and any nested pipeline manifest
        pointing here via ``parent_run_id`` — joined to a run that no consumer can
        resolve. The decision is on the payload as ``sampled`` instead, so an
        offline consumer can filter on it.
        """
        if self.finished:
            return
        self.finished = True
        try:
            hook._emit_manifest(
                KIND_RUN_MANIFEST,
                self.to_manifest(),
                payload_kind="json",
                run_id=self.run_id,
                dataset_id=self.resolved_dataset_id(),
            )
        except Exception as exc:
            hook.logger.debug("capture manifest emit failed (%s)", exc)


def current_scope() -> RunScope | None:
    """The innermost active run scope, if any."""
    return hook._current_scope.get()


@contextmanager
def run_scope(
    run_id: UUID | str | None,
    dataset_id: UUID | str | None = None,
    *,
    kind: str = "operation",
) -> Iterator[RunScope]:
    """Bracket one run; enqueue its manifest on exit when the run is sampled.

    Sampling is per run: operation scopes roll ``SAMPLE_RATE`` once here,
    pipeline scopes are always sampled. Entry also starts the running loop's
    flusher so emits from worker threads during the run are flushed on the
    interval tick even if nothing ever emits on-loop.
    """
    scope = RunScope(
        run_id=run_id,
        dataset_id=dataset_id,
        kind=kind,
        sampled=(kind != "operation") or random.random() < hook.SAMPLE_RATE,
        started_at=time.time(),
        dropped_at_enter=hook._dropped,
        parent=hook._current_scope.get(),
    )
    token = hook._current_scope.set(scope)
    hook.ensure_flusher()
    try:
        yield scope
    finally:
        scope.finish()
        # Same cross-context tolerance as operation_usage_scope: the enclosing
        # generator may be finalized from a context other than the one that
        # resumed it last, in which case reset() raises ValueError.
        try:
            hook._current_scope.reset(token)
        except ValueError:
            pass


def note(key: str, value: Any) -> None:
    """Record a manifest field on the active scope; no-op without one (~15 ns)."""
    scope = hook._current_scope.get()
    if scope is None:
        return
    scope.fields[key] = value


def bump(counter: str, n: int = 1) -> None:
    """Increment a manifest counter on the active scope; no-op without one."""
    scope = hook._current_scope.get()
    if scope is None:
        return
    scope.counters[counter] += n
