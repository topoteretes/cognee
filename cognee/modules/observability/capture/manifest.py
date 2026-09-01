"""Run manifests for eval capture (SDK-529).

A ``run_scope`` brackets one operation or pipeline run. It is a SYNC
``@contextmanager`` — the same shape as ``operation_usage_scope`` /
``parent_run_scope`` in ``cognee/modules/operations/usage_accumulator.py`` — so
it stacks on the existing ``with`` line inside the ``run_tasks`` async
generator. The ``RunScope`` is mutated IN PLACE and the contextvar is bound to
that one object before any task fan-out, so ``note()``/``bump()`` from
``create_task`` children reach the same accumulator. On exit a
``run.manifest`` event is enqueued (no drain here — callers ``await drain()``
themselves where appropriate).
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

    def set_dataset(self, dataset_id: UUID | str | None) -> None:
        """Bind the dataset late. Because ``CaptureEvent.scope`` is resolved at
        serialization time, events already buffered for this run pick it up too;
        only events flushed BEFORE this call (≤ FLUSH_INTERVAL_S window) can land
        under ``nodataset/`` — the manifest is authoritative."""
        self.dataset_id = dataset_id

    def note(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def bump(self, counter: str, n: int = 1) -> None:
        self.counters[counter] += n

    def to_manifest(self) -> dict[str, Any]:
        ended_at = time.time()
        return {
            "run_id": None if self.run_id is None else str(self.run_id),
            "dataset_id": None if self.dataset_id is None else str(self.dataset_id),
            "kind": self.kind,
            "sampled": self.sampled,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "duration_s": ended_at - self.started_at,
            **self.fields,
            "counters": dict(self.counters),
            # Process-global counter delta: an approximation under concurrent runs.
            "dropped_events": hook._dropped - self.dropped_at_enter,
        }


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
    pipeline scopes are always sampled.
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
    try:
        yield scope
    finally:
        if scope.sampled:
            try:
                hook.emit(
                    KIND_RUN_MANIFEST,
                    scope.to_manifest(),
                    payload_kind="json",
                    run_id=scope.run_id,
                    dataset_id=scope.dataset_id,
                )
            except Exception as exc:
                hook.logger.debug("capture manifest emit failed (%s)", exc)
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
