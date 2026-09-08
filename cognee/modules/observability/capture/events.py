"""Capture event record and kind constants (SDK-529).

A ``CaptureEvent`` is what ``emit()`` appends to the buffer: an object
reference (or a snapshot, see the snapshot rule in ``hook.py``) plus the
attribution needed to place it later. Nothing here serializes — the flusher
turns events into the record shape handed to sinks::

    {
        "schema_version": int,           # CAPTURE_SCHEMA_VERSION
        "event_id": str,                 # "<process>-<seq>", stable across redelivery
        "kind", "run_id": str|None, "dataset_id": str|None, "stage", "ts", "payload",
    }

where ``payload`` is ``model_dump(mode="json")`` for ``payload_kind="pydantic"``,
the object as-is for ``"json"`` (must be JSON-serializable; sinks apply
``json.dumps(default=str)``), and the str as-is for ``"text"``.

``event_id`` is assigned at emit, not at serialization: delivery is
at-least-once (a flush cut off mid-write is re-buffered and written again),
and an id minted while serializing would differ between the two copies,
defeating the deduplication it exists for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from .manifest import RunScope

# Version of the record envelope and the payload shapes under each kind. Bump
# on any change a consumer of the persisted files could not absorb silently:
# a renamed or re-typed envelope key, a payload key changing meaning, a kind
# being split. Additive payload keys do not need a bump.
CAPTURE_SCHEMA_VERSION = 1

KIND_EXTRACTION_CHUNK_GRAPH = "extraction.chunk_graph"
KIND_EXTRACTION_DROPPED_DUPLICATES = "extraction.dropped_duplicates"
KIND_EXTRACTION_FUZZY_MATCH = "extraction.fuzzy_match"
KIND_STORAGE_DELTA = "storage.delta"
KIND_SUMMARY_GENERATED = "summary.generated"
KIND_RETRIEVAL_CANDIDATES = "retrieval.candidates"
KIND_RUN_MANIFEST = "run.manifest"

# Retrieval kinds are the ones subject to per-run sampling (see should_capture).
RETRIEVAL_KIND_PREFIX = "retrieval."


@dataclass(slots=True)
class CaptureEvent:
    """One buffered observation; serialized later by the flusher, never at emit."""

    kind: str
    # Object reference (or snapshot per the snapshot rule).
    payload: Any
    # "pydantic" | "json" | "text"
    payload_kind: str
    # Explicit overrides — raw at emit, str()-coerced by the flusher.
    run_id: UUID | str | None
    dataset_id: UUID | str | None
    # The active manifest scope at emit time. The flusher resolves
    # ``run_id = event.run_id or scope.run_id`` (same for dataset_id) AT
    # SERIALIZATION TIME, so a dataset bound late via ``RunScope.set_dataset()``
    # shows up on events that were already buffered.
    scope: RunScope | None
    stage: str | None
    # time.time() at emit.
    ts: float
    # Position in this process's emit sequence; with the per-process prefix it
    # forms ``event_id``. Assigned at emit so a requeued event keeps its id.
    seq: int
    # Set by the flusher on the first sink acknowledgement, so a run's
    # ``events_delivered`` credits an event once even if at-least-once delivery
    # writes it twice.
    delivered: bool = False
