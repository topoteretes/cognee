"""Eval capture — non-blocking capture of pipeline output for offline evals (SDK-529).

Public surface is this package only. It is deliberately NOT re-exported from
``cognee.modules.observability`` (import cycle with ``cognee/base_config.py``
and import cost of ``import cognee``). Nothing under ``cognee_db_workers/``
imports it; emit points live on the adapter side in ``cognee/``.

Typical use::

    from cognee.modules.observability import capture

    if capture.is_active() and capture.should_capture(capture.KIND_EXTRACTION_CHUNK_GRAPH):
        capture.emit(
            capture.KIND_EXTRACTION_CHUNK_GRAPH,
            graph.model_dump(mode="json"),   # snapshot: the graph is mutated afterwards
            payload_kind="json",
            stage="extract_graph_from_data",
        )
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
