"""Tests for the Langfuse OTLP integration.

Covers the three pieces of the feature, all offline (no live Langfuse):
  1. ergonomic ``LANGFUSE_*`` config -> OTLP endpoint + Basic-auth header,
  2. the exporter wiring (Langfuse forces the HTTP exporter),
  3. the attribute mapping, asserted on *exported* span payloads via an
     in-memory stub collector.
"""

import os
import asyncio
import base64
from unittest.mock import patch, MagicMock

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider  # noqa: F401

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# The exporter-wiring and span-mapping tests need the cognee[tracing] extra; the
# config/header tests below do not. Mirrors the guard in test_tracing.py.
requires_otel = pytest.mark.skipif(
    not _OTEL_AVAILABLE, reason="requires the cognee[tracing] extra (opentelemetry-sdk)"
)


@pytest.fixture(autouse=True)
def _reset_base_config_cache():
    """Keep the cached, env-derived BaseConfig from leaking between tests."""
    from cognee.base_config import get_base_config

    get_base_config.cache_clear()
    yield
    get_base_config.cache_clear()


# ---------------------------------------------------------------------------
# Ergonomic config
# ---------------------------------------------------------------------------
def test_langfuse_config_builds_endpoint_and_headers():
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-12345",
            "LANGFUSE_SECRET_KEY": "sk-67890",
            "LANGFUSE_HOST": "https://us.cloud.langfuse.com",
        },
        clear=True,
    ):
        from cognee.base_config import get_base_config

        config = get_base_config()

        assert (
            config.otel_exporter_otlp_endpoint
            == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
        )
        expected = base64.b64encode(b"pk-12345:sk-67890").decode("utf-8")
        assert config.otel_exporter_otlp_headers == f"Authorization=Basic {expected}"
        assert config.cognee_tracing_enabled is True


def test_langfuse_config_requires_both_keys():
    with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-123"}, clear=True):
        from cognee.base_config import get_base_config

        with pytest.raises(ValueError, match="must be provided together"):
            get_base_config()


def test_langfuse_config_respects_explicit_endpoint():
    """An explicitly configured OTLP endpoint/headers wins over the derived one."""
    with patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-1",
            "LANGFUSE_SECRET_KEY": "sk-2",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.internal/v1/traces",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer token",
        },
        clear=True,
    ):
        from cognee.base_config import get_base_config

        config = get_base_config()

        assert config.otel_exporter_otlp_endpoint == "https://collector.internal/v1/traces"
        assert config.otel_exporter_otlp_headers == "Authorization=Bearer token"
        assert config.cognee_tracing_enabled is True


# ---------------------------------------------------------------------------
# Exporter wiring
# ---------------------------------------------------------------------------
def test_parse_otlp_headers():
    from cognee.modules.observability.tracing import _parse_otlp_headers

    assert _parse_otlp_headers(None) is None
    assert _parse_otlp_headers("") is None
    # Splitting on the first "=" preserves base64 "==" padding in the value.
    assert _parse_otlp_headers("Authorization=Basic YWJjOg==") == {
        "Authorization": "Basic YWJjOg=="
    }
    assert _parse_otlp_headers("a=1, b=2") == {"a": "1", "b": "2"}


@requires_otel
def test_langfuse_endpoint_forces_http_exporter(monkeypatch):
    """Langfuse endpoints route to the HTTP exporter (gRPC is unsupported), built with
    the derived endpoint + Basic-auth header. A fake exporter module stands in so the
    test does not require the optional opentelemetry-exporter-otlp-proto-http package."""
    import sys
    import types

    with patch.dict(
        os.environ,
        {"LANGFUSE_PUBLIC_KEY": "pk-12345", "LANGFUSE_SECRET_KEY": "sk-67890"},
        clear=True,
    ):
        from cognee.base_config import get_base_config
        from cognee.modules.observability.tracing import _try_add_otlp_exporter

        get_base_config.cache_clear()

        constructed = {}

        class FakeHttpExporter:
            def __init__(self, endpoint=None, headers=None):
                constructed["endpoint"] = endpoint
                constructed["headers"] = headers

        module_name = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        fake_module = types.ModuleType(module_name)
        fake_module.OTLPSpanExporter = FakeHttpExporter
        monkeypatch.setitem(sys.modules, module_name, fake_module)

        provider = MagicMock()
        _try_add_otlp_exporter(provider)

        assert constructed["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
        expected = base64.b64encode(b"pk-12345:sk-67890").decode("utf-8")
        assert constructed["headers"]["Authorization"] == f"Basic {expected}"
        provider.add_span_processor.assert_called_once()


# ---------------------------------------------------------------------------
# Attribute mapping — asserted on exported span payloads
# ---------------------------------------------------------------------------
def _tracer_with_collector():
    """A real TracerProvider wired to an in-memory stub OTLP collector."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    collector = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(collector))
    return collector, provider, provider.get_tracer("test")


def _run_observed(monkeypatch, category, adapter_cls):
    """Drive an @observe-decorated method with tracing pointed at a stub collector,
    and return the single exported span."""
    import cognee.modules.observability.tracing as tracing
    import cognee.modules.observability.trace_context as trace_context
    from cognee.modules.observability.get_observe import get_observe

    collector, provider, tracer = _tracer_with_collector()
    monkeypatch.setattr(tracing, "_tracer", tracer)
    monkeypatch.setattr(trace_context, "_tracing_enabled", True)

    observe = get_observe()

    class _Decorated(adapter_cls):
        @observe(as_type=category)
        async def run(self):
            return "ok"

    assert asyncio.run(_Decorated().run()) == "ok"
    provider.force_flush()

    spans = collector.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


@requires_otel
def test_generation_span_exports_genai_attributes(monkeypatch):
    from opentelemetry.trace import SpanKind

    class OpenAILike:
        model = "gpt-4o-mini"
        name = "OpenAI"

    span = _run_observed(monkeypatch, "generation", OpenAILike)
    attrs = dict(span.attributes)

    assert span.kind == SpanKind.CLIENT
    assert attrs["cognee.span.category"] == "generation"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.system"] == "openai"  # lowercased per GenAI convention
    assert attrs["langfuse.observation.type"] == "generation"


@requires_otel
def test_generation_span_without_model_is_still_marked(monkeypatch):
    """llama.cpp local mode leaves model=None; the span must still classify as a
    generation via the langfuse.observation.type marker."""

    class LocalLike:
        model = None
        name = "LlamaCpp"

    span = _run_observed(monkeypatch, "generation", LocalLike)
    attrs = dict(span.attributes)

    assert "gen_ai.request.model" not in attrs  # None is never written
    assert attrs["gen_ai.system"] == "llamacpp"
    assert attrs["langfuse.observation.type"] == "generation"


@requires_otel
def test_non_generation_span_has_no_genai_attributes(monkeypatch):
    from opentelemetry.trace import SpanKind

    class Anything:
        model = "should-be-ignored"
        name = "OpenAI"

    span = _run_observed(monkeypatch, "transcription", Anything)
    attrs = dict(span.attributes)

    assert span.kind == SpanKind.INTERNAL
    assert attrs["cognee.span.category"] == "transcription"
    assert "gen_ai.request.model" not in attrs
    assert "langfuse.observation.type" not in attrs
