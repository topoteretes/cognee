from typing import Optional
from contextlib import contextmanager

from .trace_context import (
    enable_tracing,
    disable_tracing,
    is_tracing_enabled,
    get_last_trace,
    get_all_traces,
    clear_traces,
)
from .tracing import (
    CogneeTrace,
    redact_secrets,
    get_tracer,
    COGNEE_DB_SYSTEM,
    COGNEE_DB_QUERY,
    COGNEE_DB_ROW_COUNT,
    COGNEE_LLM_MODEL,
    COGNEE_LLM_PROVIDER,
    COGNEE_SEARCH_TYPE,
    COGNEE_SEARCH_QUERY,
    COGNEE_PIPELINE_TASK_NAME,
    COGNEE_VECTOR_COLLECTION,
    COGNEE_VECTOR_RESULT_COUNT,
    COGNEE_SPAN_CATEGORY,
    COGNEE_PIPELINE_NAME,
    COGNEE_RESULT_SUMMARY,
    COGNEE_RESULT_COUNT,
)


try:
    from opentelemetry.trace import StatusCode as OtelStatusCode
except ImportError:

    class _StatusCodeFallback:
        ERROR = "ERROR"
        OK = "OK"
        UNSET = "UNSET"

    OtelStatusCode = _StatusCodeFallback  # type: ignore[misc, assignment]


class _NullSpan:
    """No-op span used when tracing is disabled."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def get_tracer_if_enabled() -> Optional[object]:
    """Return the OTEL tracer if tracing is enabled, None otherwise."""
    if is_tracing_enabled():
        return get_tracer()
    return None


@contextmanager
def new_span(name: str):
    """Context manager that creates an OTEL span if tracing is enabled, or yields None."""
    if is_tracing_enabled():
        tracer = get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span(name) as span:
                yield span
                return
    yield _NullSpan()
