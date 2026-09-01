"""Eval capture — non-blocking capture of pipeline output for offline evals (SDK-529).

Public surface is this package only. It is deliberately NOT re-exported from
``cognee.modules.observability`` (import cycle with ``cognee/base_config.py``
and import cost of ``import cognee``). Nothing under ``cognee_db_workers/``
imports it; emit points live on the adapter side in ``cognee/``.

Typical use - the ``extraction.chunk_graph`` emit point in ``extract_graph_from_data``
(``should_capture`` only gates the sampled ``retrieval.*`` kinds; it is True for the
rest, so pipeline emit points check ``is_active()`` alone)::

    from cognee.modules.observability import capture

    if capture.is_active():
        capture.emit(
            capture.KIND_EXTRACTION_CHUNK_GRAPH,
            {
                "chunk_id": str(chunk.id),
                "chunk_index": chunk_index,
                "chunk_size_chars": len(chunk.text),
                # A snapshot taken now, never the live object or a model_copy:
                # the graph is mutated (dedup, canonicalization) afterwards.
                "graph": graph.model_dump(mode="json"),
            },
            payload_kind="json",
            stage="extract_graph_from_data",
        )

Payloads are plain JSON (dicts/lists of scalars) or back-reference-free pydantic
models - never graph elements, scored results or DataPoint instances. Consumers
read the chunk graph at ``payload["graph"]`` next to its join keys.
"""

from .config import CaptureConfig, get_capture_config
from .events import (
    KIND_EXTRACTION_CHUNK_GRAPH,
    KIND_EXTRACTION_DROPPED_DUPLICATES,
    KIND_EXTRACTION_FUZZY_MATCH,
    KIND_RETRIEVAL_CANDIDATES,
    KIND_RUN_MANIFEST,
    KIND_STORAGE_DELTA,
    KIND_SUMMARY_GENERATED,
    RETRIEVAL_KIND_PREFIX,
    CaptureEvent,
)
from .hook import drain, emit, is_active, register_capture_sink, should_capture, shutdown
from .manifest import RunScope, bump, current_scope, note, run_scope
from .prompt_hash import prompt_file_fingerprint, prompt_fingerprint
from .sinks import CaptureSink, StorageSink

__all__ = [
    "CaptureConfig",
    "CaptureEvent",
    "CaptureSink",
    "KIND_EXTRACTION_CHUNK_GRAPH",
    "KIND_EXTRACTION_DROPPED_DUPLICATES",
    "KIND_EXTRACTION_FUZZY_MATCH",
    "KIND_RETRIEVAL_CANDIDATES",
    "KIND_RUN_MANIFEST",
    "KIND_STORAGE_DELTA",
    "KIND_SUMMARY_GENERATED",
    "RETRIEVAL_KIND_PREFIX",
    "RunScope",
    "StorageSink",
    "bump",
    "current_scope",
    "drain",
    "emit",
    "get_capture_config",
    "is_active",
    "note",
    "prompt_file_fingerprint",
    "prompt_fingerprint",
    "register_capture_sink",
    "run_scope",
    "should_capture",
    "shutdown",
]
