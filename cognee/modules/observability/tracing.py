"""OpenTelemetry-native tracing for Cognee.

Provides CogneeSpanExporter (in-memory span buffer), setup_tracing(),
get_tracer(), CogneeTrace, semantic attribute constants, and redaction.
"""

import os
import re
import threading
from collections import defaultdict
from typing import Optional, Sequence

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
    from opentelemetry.sdk.trace.export import (
        SpanExporter,
        SpanExportResult,
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Semantic attribute constants
# ---------------------------------------------------------------------------
COGNEE_DB_SYSTEM = "cognee.db.system"
COGNEE_DB_QUERY = "cognee.db.query"
COGNEE_DB_ROW_COUNT = "cognee.db.row_count"
COGNEE_LLM_MODEL = "cognee.llm.model"
COGNEE_LLM_PROVIDER = "cognee.llm.provider"
COGNEE_SEARCH_TYPE = "cognee.search.type"
COGNEE_SEARCH_QUERY = "cognee.search.query"
COGNEE_PIPELINE_TASK_NAME = "cognee.pipeline.task_name"
COGNEE_VECTOR_COLLECTION = "cognee.vector.collection"
COGNEE_VECTOR_RESULT_COUNT = "cognee.vector.result_count"
COGNEE_SPAN_CATEGORY = "cognee.span.category"
COGNEE_RESULT_SUMMARY = "cognee.result.summary"
COGNEE_RESULT_COUNT = "cognee.result.count"
COGNEE_PIPELINE_NAME = "cognee.pipeline.name"

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
    re.compile(r"(api[_-]?key\s*[=:]\s*)['\"]?[A-Za-z0-9\-_]{16,}['\"]?", re.IGNORECASE),
    re.compile(r"(bearer\s+)[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE),
    re.compile(r"(password\s*[=:]\s*)['\"]?[^\s'\"]{8,}['\"]?", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Redact common API key and secret patterns from text."""
    if not text:
        return text
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(lambda m: m.group(0)[:6] + "***REDACTED***", result)
    return result


def _check_otel_available() -> None:
    if not _OTEL_AVAILABLE:
        raise ImportError(
            "OpenTelemetry packages are required for tracing. "
            "Install them with: pip install cognee[tracing]"
        )


# ---------------------------------------------------------------------------
# In-memory span exporter
# ---------------------------------------------------------------------------
_MAX_TRACES = 50

if _OTEL_AVAILABLE:

    class CogneeSpanExporter(SpanExporter):
        """Custom SpanExporter that buffers completed spans in-memory.

        Spans are grouped by trace_id. The buffer is bounded to the last
        ``_MAX_TRACES`` distinct traces.
        """

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._traces: dict[str, list[dict]] = defaultdict(list)
            self._trace_order: list[str] = []

        def export(self, spans: Sequence["ReadableSpan"]) -> "SpanExportResult":
            with self._lock:
                for span in spans:
                    trace_id = format(span.context.trace_id, "032x")
                    span_dict = {
                        "name": span.name,
                        "trace_id": trace_id,
                        "span_id": format(span.context.span_id, "016x"),
                        "parent_span_id": (
                            format(span.parent.span_id, "016x") if span.parent else None
                        ),
                        "start_time_ns": span.start_time,
                        "end_time_ns": span.end_time,
                        "duration_ms": (
                            (span.end_time - span.start_time) / 1_000_000
                            if span.end_time and span.start_time
                            else 0.0
                        ),
                        "status": span.status.status_code.name if span.status else "UNSET",
                        "attributes": dict(span.attributes) if span.attributes else {},
                    }
                    self._traces[trace_id].append(span_dict)
                    if trace_id not in self._trace_order:
                        self._trace_order.append(trace_id)

                # Evict oldest traces if over limit
                while len(self._trace_order) > _MAX_TRACES:
                    oldest = self._trace_order.pop(0)
                    self._traces.pop(oldest, None)

            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

        # -- Public helpers for reading collected traces --

        def get_last_trace_spans(self) -> Optional[list[dict]]:
            with self._lock:
                if not self._trace_order:
                    return None
                last_id = self._trace_order[-1]
                return list(self._traces[last_id])

        def get_all_traces(self) -> dict[str, list[dict]]:
            with self._lock:
                return {tid: list(spans) for tid, spans in self._traces.items()}

        def clear(self) -> None:
            with self._lock:
                self._traces.clear()
                self._trace_order.clear()


# ---------------------------------------------------------------------------
# CogneeTrace -- lightweight view over collected spans
# ---------------------------------------------------------------------------
class CogneeTrace:
    """Lightweight read-only view over spans belonging to a single trace."""

    def __init__(self, span_list: list[dict]) -> None:
        self._spans = span_list

    def spans(self) -> list[dict]:
        """Return flat list of span dicts sorted by start time."""
        return sorted(self._spans, key=lambda s: s.get("start_time_ns", 0))

    def summary(self) -> dict:
        """Return summary with operation, total_duration_ms, breakdown, errors."""
        if not self._spans:
            return {
                "operation": "",
                "total_duration_ms": 0.0,
                "span_count": 0,
                "breakdown": {},
                "errors": [],
            }

        sorted_spans = self.spans()

        # Find root span (no parent)
        root = next((s for s in sorted_spans if s.get("parent_span_id") is None), sorted_spans[0])

        # Build breakdown by span name prefix (group by base name)
        breakdown: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
        errors: list[str] = []

        for s in sorted_spans:
            name = s["name"]
            breakdown[name]["count"] += 1
            breakdown[name]["total_ms"] += s.get("duration_ms", 0.0)
            if s.get("status") == "ERROR":
                errors.append(f"{name}: error")

        return {
            "operation": root["name"],
            "total_duration_ms": root.get("duration_ms", 0.0),
            "span_count": len(sorted_spans),
            "breakdown": dict(breakdown),
            "errors": errors,
        }

    def tree(self) -> dict:
        """Return hierarchical span tree as nested dicts."""
        spans_by_id = {s["span_id"]: {**s, "children": []} for s in self._spans}
        roots = []

        for s in self._spans:
            node = spans_by_id[s["span_id"]]
            parent_id = s.get("parent_span_id")
            if parent_id and parent_id in spans_by_id:
                spans_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)

        return roots[0] if len(roots) == 1 else {"children": roots}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_exporter: Optional["CogneeSpanExporter"] = None
_tracer: Optional["trace.Tracer"] = None
_provider: Optional["TracerProvider"] = None


def setup_tracing(console_output: bool = False) -> "trace.Tracer":
    """Create an OTEL TracerProvider with CogneeSpanExporter.

    Optionally adds a ConsoleSpanExporter for debugging.
    Returns the cognee tracer.
    """
    _check_otel_available()

    from cognee.version import get_cognee_version

    global _exporter, _tracer, _provider

    version = get_cognee_version()
    resource = Resource.create(
        {
            "service.name": "cognee",
            "service.version": version,
            "deployment.environment": os.getenv("ENV", "development"),
        }
    )

    _exporter = CogneeSpanExporter()
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(SimpleSpanProcessor(_exporter))

    if console_output:
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("cognee", version)
    return _tracer


def get_tracer() -> Optional["trace.Tracer"]:
    """Return the cognee OTEL tracer, or None if not set up."""
    return _tracer


def get_provider() -> Optional["TracerProvider"]:
    """Return the TracerProvider (useful for adding external exporters)."""
    return _provider


def get_exporter() -> Optional["CogneeSpanExporter"]:
    """Return the in-memory exporter."""
    return _exporter


def shutdown_tracing(timeout_ms: int = 30000) -> None:
    """Shut down the TracerProvider and clear global state."""
    global _exporter, _tracer, _provider

    if _provider is not None:
        _provider.force_flush(timeout_millis=timeout_ms)
        _provider.shutdown()
    _exporter = None
    _tracer = None
    _provider = None
