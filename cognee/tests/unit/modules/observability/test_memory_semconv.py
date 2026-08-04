"""Tests for memory-semconv v0.1.0 OTel instrumentation.

Validates that the core memory operations emit correct spans (with memory.*
attributes), metrics, and that the log bridge attaches without error.
"""

import time
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_otel_state():
    """Reset OTel providers between tests to avoid cross-contamination."""
    yield
    try:
        from cognee.modules.observability.trace_context import disable_tracing

        disable_tracing()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Span / attribute tests
# ---------------------------------------------------------------------------


def test_memory_semconv_constants_exported():
    """memory-semconv attribute constants are exported from the observability package."""
    from cognee.modules.observability import (
        MEMORY_SYSTEM,
        MEMORY_OPERATION,
        MEMORY_QUERY_TEXT,
        MEMORY_QUERY_TYPE,
        MEMORY_RESULT_COUNT,
        MEMORY_DATA_SIZE,
        MEMORY_ITEM_COUNT,
        MEMORY_COLLECTION,
        MEMORY_STORE_BACKEND,
    )

    assert MEMORY_SYSTEM == "memory.system"
    assert MEMORY_OPERATION == "memory.operation"
    assert MEMORY_QUERY_TEXT == "memory.query.text"
    assert MEMORY_QUERY_TYPE == "memory.query.type"
    assert MEMORY_RESULT_COUNT == "memory.result.count"
    assert MEMORY_DATA_SIZE == "memory.data.size"
    assert MEMORY_ITEM_COUNT == "memory.item.count"
    assert MEMORY_COLLECTION == "memory.collection"
    assert MEMORY_STORE_BACKEND == "memory.store.backend"


def test_enable_tracing_sets_up_all_signals():
    """enable_tracing() initialises tracing, metrics, and log bridge without error."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.trace_context import (
        enable_tracing,
        disable_tracing,
        is_tracing_enabled,
    )

    enable_tracing()
    assert is_tracing_enabled()

    from cognee.modules.observability.metrics import get_meter

    assert get_meter() is not None

    disable_tracing()


def test_new_span_emits_memory_operation_retrieve():
    """A search-style span tagged with memory.operation=retrieve is collected."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.trace_context import enable_tracing
    from cognee.modules.observability import (
        new_span,
        MEMORY_SYSTEM,
        MEMORY_OPERATION,
        MEMORY_QUERY_TEXT,
        MEMORY_RESULT_COUNT,
    )
    from cognee.modules.observability.trace_context import get_last_trace

    enable_tracing()

    with new_span("memory.retrieve") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "retrieve")
        span.set_attribute(MEMORY_QUERY_TEXT, "test query"[:500])
        span.set_attribute(MEMORY_RESULT_COUNT, 3)

    trace = get_last_trace()
    assert trace is not None, "Expected a trace to be recorded"

    spans = trace.spans()
    root = spans[0]
    assert root["name"] == "memory.retrieve"
    attrs = root["attributes"]
    assert attrs.get("memory.system") == "cognee"
    assert attrs.get("memory.operation") == "retrieve"
    assert attrs.get("memory.query.text") == "test query"
    assert attrs.get("memory.result.count") == 3


def test_new_span_emits_memory_operation_store():
    """A store span tagged with memory.operation=store is collected."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.trace_context import enable_tracing
    from cognee.modules.observability import (
        new_span,
        MEMORY_SYSTEM,
        MEMORY_OPERATION,
        MEMORY_COLLECTION,
    )
    from cognee.modules.observability.trace_context import get_last_trace

    enable_tracing()

    with new_span("memory.store") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "store")
        span.set_attribute(MEMORY_COLLECTION, "my_dataset")

    trace = get_last_trace()
    assert trace is not None
    spans = trace.spans()
    assert spans[0]["attributes"].get("memory.collection") == "my_dataset"


def test_new_span_emits_memory_operation_delete():
    """A delete span tagged with memory.operation=delete is collected."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.trace_context import enable_tracing
    from cognee.modules.observability import new_span, MEMORY_SYSTEM, MEMORY_OPERATION
    from cognee.modules.observability.trace_context import get_last_trace

    enable_tracing()

    with new_span("memory.delete") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "delete")

    trace = get_last_trace()
    assert trace is not None
    spans = trace.spans()
    assert spans[0]["name"] == "memory.delete"
    assert spans[0]["attributes"].get("memory.operation") == "delete"


def test_new_span_emits_memory_operation_process():
    """A process span tagged with memory.operation=process is collected."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.trace_context import enable_tracing
    from cognee.modules.observability import new_span, MEMORY_SYSTEM, MEMORY_OPERATION
    from cognee.modules.observability.trace_context import get_last_trace

    enable_tracing()

    with new_span("memory.process") as span:
        span.set_attribute(MEMORY_SYSTEM, "cognee")
        span.set_attribute(MEMORY_OPERATION, "process")

    trace = get_last_trace()
    assert trace is not None
    spans = trace.spans()
    assert spans[0]["name"] == "memory.process"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metric_helpers_are_no_ops_before_setup():
    """Metric helpers do not raise even when metrics are not configured."""
    from cognee.modules.observability import (
        record_operation_duration,
        increment_items_stored,
        increment_items_retrieved,
        increment_items_deleted,
        record_query_results,
        increment_bytes_stored,
        increment_vector_searches,
        increment_graph_edges,
        increment_graph_nodes,
        increment_operation_errors,
    )

    attrs = {"memory.system": "cognee", "memory.operation": "store"}
    record_operation_duration(42.0, attrs)
    increment_items_stored(5, attrs)
    increment_items_retrieved(3, attrs)
    increment_items_deleted(1, attrs)
    record_query_results(2, attrs)
    increment_bytes_stored(1024, attrs)
    increment_vector_searches(attrs)
    increment_graph_edges(10, attrs)
    increment_graph_nodes(5, attrs)
    increment_operation_errors(attrs)


def test_metric_names_follow_semconv():
    """Exported metric name constants match memory-semconv v0.1.0."""
    from cognee.modules.observability import (
        MEMORY_OPERATION_DURATION,
        MEMORY_ITEMS_STORED,
        MEMORY_ITEMS_RETRIEVED,
        MEMORY_ITEMS_DELETED,
        MEMORY_QUERY_RESULT_COUNT,
        MEMORY_DATA_BYTES_STORED,
        MEMORY_VECTOR_SEARCHES,
        MEMORY_GRAPH_EDGES_ADDED,
        MEMORY_GRAPH_NODES_ADDED,
        MEMORY_OPERATION_ERRORS,
    )

    assert MEMORY_OPERATION_DURATION == "memory.operation.duration"
    assert MEMORY_ITEMS_STORED == "memory.items.stored"
    assert MEMORY_ITEMS_RETRIEVED == "memory.items.retrieved"
    assert MEMORY_ITEMS_DELETED == "memory.items.deleted"
    assert MEMORY_QUERY_RESULT_COUNT == "memory.query.result.count"
    assert MEMORY_DATA_BYTES_STORED == "memory.data.bytes.stored"
    assert MEMORY_VECTOR_SEARCHES == "memory.vector.searches"
    assert MEMORY_GRAPH_EDGES_ADDED == "memory.graph.edges.added"
    assert MEMORY_GRAPH_NODES_ADDED == "memory.graph.nodes.added"
    assert MEMORY_OPERATION_ERRORS == "memory.operation.errors"


def test_setup_metrics_returns_meter():
    """setup_metrics() returns a meter when the OTel SDK is available."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.metrics import setup_metrics, get_meter, shutdown_metrics

    meter = setup_metrics()
    assert meter is not None
    assert get_meter() is meter

    shutdown_metrics()
    assert get_meter() is None


def test_metric_helpers_accept_calls_after_setup():
    """Metric helpers do not raise after metrics are configured."""
    pytest.importorskip("opentelemetry")

    from cognee.modules.observability.metrics import setup_metrics, shutdown_metrics
    from cognee.modules.observability import (
        record_operation_duration,
        record_query_results,
        increment_items_stored,
        increment_items_retrieved,
    )

    setup_metrics()
    attrs = {"memory.system": "cognee", "memory.operation": "retrieve"}
    record_operation_duration(10.5, attrs)
    record_query_results(7, attrs)
    increment_items_stored(3, {"memory.system": "cognee", "memory.operation": "store"})
    increment_items_retrieved(7, attrs)
    shutdown_metrics()


# ---------------------------------------------------------------------------
# Log bridge tests
# ---------------------------------------------------------------------------


def test_log_bridge_attaches_without_error():
    """setup_log_bridge() attaches an OTel handler to cognee loggers."""
    pytest.importorskip("opentelemetry")

    import logging
    from cognee.modules.observability.logs import setup_log_bridge, shutdown_log_bridge

    setup_log_bridge()

    cognee_logger = logging.getLogger("cognee")
    has_otel_handler = any(type(h).__name__ == "LoggingHandler" for h in cognee_logger.handlers)
    assert has_otel_handler, "Expected OTel LoggingHandler on cognee logger"

    # Logging through the bridge should not raise
    cognee_logger.info("OTel log bridge test message")

    shutdown_log_bridge()

    has_otel_handler_after = any(
        type(h).__name__ == "LoggingHandler" for h in cognee_logger.handlers
    )
    assert not has_otel_handler_after, "Expected handler removed after shutdown"


def test_enable_tracing_wires_log_bridge():
    """enable_tracing() causes cognee logger to gain an OTel handler."""
    pytest.importorskip("opentelemetry")

    import logging
    from cognee.modules.observability.trace_context import enable_tracing, disable_tracing

    enable_tracing()
    cognee_logger = logging.getLogger("cognee")
    has_otel = any(type(h).__name__ == "LoggingHandler" for h in cognee_logger.handlers)
    assert has_otel

    disable_tracing()


# ---------------------------------------------------------------------------
# Dynatrace / HTTP-only exporter detection tests
# ---------------------------------------------------------------------------


def test_requires_http_exporter_dynatrace():
    """Dynatrace endpoints are correctly identified as HTTP-only."""
    from cognee.modules.observability.tracing import _requires_http_exporter

    assert _requires_http_exporter("https://abc12345.live.dynatrace.com/api/v2/otlp/v1/traces")
    assert _requires_http_exporter("https://otel.live.dynatrace.com/api/v2/otlp")
    assert _requires_http_exporter("https://myenv.dynatrace.com/api/v2/otlp/v1/traces")


def test_requires_http_exporter_langfuse():
    """Langfuse endpoints are correctly identified as HTTP-only."""
    from cognee.modules.observability.tracing import _requires_http_exporter

    assert _requires_http_exporter("https://cloud.langfuse.com/api/public/otel")
    assert _requires_http_exporter("https://us.cloud.langfuse.com/api/public/otel/v1/traces")


def test_requires_http_exporter_port_4318():
    """Port 4318 is the standard OTLP HTTP port and must force HTTP."""
    from cognee.modules.observability.tracing import _requires_http_exporter

    assert _requires_http_exporter("http://collector.example.com:4318")


def test_requires_http_exporter_grpc_passes():
    """Standard gRPC endpoints (port 4317) do not trigger HTTP-only mode."""
    from cognee.modules.observability.tracing import _requires_http_exporter

    # Standard gRPC collector should NOT be forced to HTTP
    assert not _requires_http_exporter("http://otel-collector.example.com:4317")
    assert not _requires_http_exporter("http://localhost:4317")
