"""Unit tests for the cognee tracing / observability module."""

import os
import pytest

from cognee.modules.observability.tracing import (
    CogneeSpanExporter,
    CogneeTrace,
    redact_secrets,
    setup_tracing,
    shutdown_tracing,
    get_tracer,
    get_exporter,
)
from cognee.modules.observability.trace_context import (
    enable_tracing,
    disable_tracing,
    is_tracing_enabled,
    get_last_trace,
    get_all_traces,
    clear_traces,
)


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------


class TestRedactSecrets:
    def test_redacts_openai_key(self):
        text = "key is sk-abcdefghijklmnopqrstuvwxyz"
        result = redact_secrets(text)
        assert "sk-abc" in result
        assert "REDACTED" in result
        assert "abcdefghijklmnopqrstuvwxyz" not in result

    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.long_token"
        result = redact_secrets(text)
        assert "REDACTED" in result

    def test_preserves_normal_text(self):
        text = "SELECT * FROM users WHERE name = 'Alice'"
        assert redact_secrets(text) == text

    def test_handles_empty_string(self):
        assert redact_secrets("") == ""

    def test_handles_none(self):
        assert redact_secrets(None) is None


# ---------------------------------------------------------------------------
# CogneeSpanExporter tests
# ---------------------------------------------------------------------------


class TestCogneeSpanExporter:
    def test_export_and_retrieve(self):
        """Test that spans are collected and retrievable."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        exporter = CogneeSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test_span") as span:
            span.set_attribute("key", "value")

        provider.force_flush()

        spans = exporter.get_last_trace_spans()
        assert spans is not None
        assert len(spans) == 1
        assert spans[0]["name"] == "test_span"
        assert spans[0]["attributes"]["key"] == "value"
        assert spans[0]["duration_ms"] >= 0

        provider.shutdown()

    def test_clear(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        exporter = CogneeSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("span1"):
            pass

        provider.force_flush()
        assert exporter.get_last_trace_spans() is not None

        exporter.clear()
        assert exporter.get_last_trace_spans() is None

        provider.shutdown()

    def test_multiple_traces(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry import context as otel_context
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        exporter = CogneeSpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        # Create two separate root spans (separate traces)
        with tracer.start_as_current_span("trace1_root"):
            pass

        # Detach context to start fresh trace
        token = otel_context.attach(otel_context.Context())
        try:
            with tracer.start_as_current_span("trace2_root"):
                pass
        finally:
            otel_context.detach(token)

        provider.force_flush()

        all_traces = exporter.get_all_traces()
        assert len(all_traces) >= 1  # At least one trace

        provider.shutdown()


# ---------------------------------------------------------------------------
# CogneeTrace tests
# ---------------------------------------------------------------------------


class TestCogneeTrace:
    def test_summary_empty(self):
        trace = CogneeTrace([])
        summary = trace.summary()
        assert summary["operation"] == ""
        assert summary["total_duration_ms"] == 0.0
        assert summary["span_count"] == 0

    def test_summary_with_spans(self):
        spans = [
            {
                "name": "cognee.search",
                "trace_id": "abc123",
                "span_id": "span1",
                "parent_span_id": None,
                "start_time_ns": 1000000000,
                "end_time_ns": 2000000000,
                "duration_ms": 1000.0,
                "status": "OK",
                "attributes": {},
            },
            {
                "name": "cognee.graph.query",
                "trace_id": "abc123",
                "span_id": "span2",
                "parent_span_id": "span1",
                "start_time_ns": 1100000000,
                "end_time_ns": 1200000000,
                "duration_ms": 100.0,
                "status": "OK",
                "attributes": {"cognee.db.system": "kuzu"},
            },
        ]

        trace = CogneeTrace(spans)
        summary = trace.summary()
        assert summary["operation"] == "cognee.search"
        assert summary["total_duration_ms"] == 1000.0
        assert summary["span_count"] == 2
        assert "cognee.graph.query" in summary["breakdown"]
        assert summary["breakdown"]["cognee.graph.query"]["count"] == 1

    def test_spans_sorted(self):
        spans = [
            {
                "name": "b",
                "span_id": "s2",
                "parent_span_id": None,
                "start_time_ns": 2000,
                "end_time_ns": 3000,
                "duration_ms": 1.0,
                "status": "OK",
                "attributes": {},
                "trace_id": "t1",
            },
            {
                "name": "a",
                "span_id": "s1",
                "parent_span_id": None,
                "start_time_ns": 1000,
                "end_time_ns": 2000,
                "duration_ms": 1.0,
                "status": "OK",
                "attributes": {},
                "trace_id": "t1",
            },
        ]
        trace = CogneeTrace(spans)
        sorted_spans = trace.spans()
        assert sorted_spans[0]["name"] == "a"
        assert sorted_spans[1]["name"] == "b"

    def test_tree(self):
        spans = [
            {
                "name": "root",
                "span_id": "s1",
                "parent_span_id": None,
                "start_time_ns": 1000,
                "end_time_ns": 3000,
                "duration_ms": 2.0,
                "status": "OK",
                "attributes": {},
                "trace_id": "t1",
            },
            {
                "name": "child",
                "span_id": "s2",
                "parent_span_id": "s1",
                "start_time_ns": 1500,
                "end_time_ns": 2500,
                "duration_ms": 1.0,
                "status": "OK",
                "attributes": {},
                "trace_id": "t1",
            },
        ]
        trace = CogneeTrace(spans)
        tree = trace.tree()
        assert tree["name"] == "root"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "child"

    def test_summary_errors(self):
        spans = [
            {
                "name": "failed_op",
                "span_id": "s1",
                "parent_span_id": None,
                "start_time_ns": 1000,
                "end_time_ns": 2000,
                "duration_ms": 1.0,
                "status": "ERROR",
                "attributes": {},
                "trace_id": "t1",
            },
        ]
        trace = CogneeTrace(spans)
        summary = trace.summary()
        assert len(summary["errors"]) == 1


# ---------------------------------------------------------------------------
# Enable/disable lifecycle
# ---------------------------------------------------------------------------


class TestTracingLifecycle:
    def setup_method(self):
        """Ensure clean state before each test."""
        disable_tracing()
        os.environ.pop("COGNEE_TRACING_ENABLED", None)

    def teardown_method(self):
        disable_tracing()
        os.environ.pop("COGNEE_TRACING_ENABLED", None)

    def test_disabled_by_default(self):
        assert not is_tracing_enabled()

    def test_enable_disable(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        enable_tracing()
        assert is_tracing_enabled()

        disable_tracing()
        assert not is_tracing_enabled()

    def test_env_var_enables(self):
        os.environ["COGNEE_TRACING_ENABLED"] = "true"
        assert is_tracing_enabled()

    def test_get_last_trace_when_disabled(self):
        assert get_last_trace() is None

    def test_get_all_traces_when_disabled(self):
        assert get_all_traces() == []

    def test_clear_traces_when_disabled(self):
        # Should not raise
        clear_traces()

    def test_full_lifecycle(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        enable_tracing()
        tracer = get_tracer()
        assert tracer is not None

        with tracer.start_as_current_span("test_op") as span:
            span.set_attribute("test_key", "test_val")

        trace = get_last_trace()
        assert trace is not None
        summary = trace.summary()
        assert summary["operation"] == "test_op"
        assert summary["span_count"] == 1

        clear_traces()
        assert get_last_trace() is None

        disable_tracing()
        assert get_tracer() is None


# ---------------------------------------------------------------------------
# Setup/shutdown
# ---------------------------------------------------------------------------


class TestSetupTracing:
    def teardown_method(self):
        shutdown_tracing()

    def test_setup_returns_tracer(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        tracer = setup_tracing()
        assert tracer is not None

    def test_setup_with_console(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        tracer = setup_tracing(console_output=True)
        assert tracer is not None

    def test_exporter_available_after_setup(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        setup_tracing()
        assert get_exporter() is not None
