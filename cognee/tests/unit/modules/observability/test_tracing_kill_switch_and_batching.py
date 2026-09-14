"""COGNEE_TRACING_ENABLED=false must be authoritative, and OTLP log export
must batch on a background thread.

Two production defects pinned here: Langfuse keys used to force
``cognee_tracing_enabled = True`` over an explicit false, leaving no working
off-switch; and the OTLP log exporters were attached through
``SimpleLogRecordProcessor``, one synchronous network round trip per log
record on the emitting thread — which silently collapses cognify throughput
under its six-figure log volume.
"""

import importlib
from unittest.mock import patch

import pytest

import cognee.base_config as base_config_module
from cognee.base_config import BaseConfig, _tracing_explicitly_disabled


def _fresh_config(monkeypatch, env: dict) -> BaseConfig:
    for key in ("COGNEE_TRACING_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return BaseConfig(
        langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY"),
    )


def test_langfuse_keys_auto_enable_when_flag_unset(monkeypatch):
    config = _fresh_config(
        monkeypatch, {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}
    )
    assert config.cognee_tracing_enabled is True
    assert "/api/public/otel" in config.otel_exporter_otlp_endpoint


def test_explicit_false_beats_langfuse_keys(monkeypatch):
    config = _fresh_config(
        monkeypatch,
        {
            "COGNEE_TRACING_ENABLED": "false",
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        },
    )
    assert config.cognee_tracing_enabled is False
    # The OTLP endpoint derivation itself is fine to keep — only the enable
    # flag is vetoed.
    assert _tracing_explicitly_disabled() is True


def test_is_tracing_enabled_vetoed_by_explicit_false(monkeypatch):
    from cognee.modules.observability import trace_context

    monkeypatch.setenv("COGNEE_TRACING_ENABLED", "false")
    # Even a latched enabled state must not survive the explicit veto.
    monkeypatch.setattr(trace_context, "_tracing_enabled", True)
    assert trace_context.is_tracing_enabled() is False


def test_otlp_log_exporters_use_batch_processor():
    """Both OTLP log exporter paths must attach a BatchLogRecordProcessor —
    a Simple processor exports synchronously per record on the emitting
    thread."""
    logs = importlib.import_module("cognee.modules.observability.logs")
    if not logs._OTEL_LOGS_AVAILABLE:
        pytest.skip("OTel logs SDK not installed")

    added = []

    class FakeProvider:
        def add_log_record_processor(self, processor):
            added.append(processor)

    try:
        importlib.import_module("opentelemetry.exporter.otlp.proto.http._log_exporter")
    except ImportError:
        pytest.skip("OTLP exporter not installed")

    logs._add_http_log_exporter(FakeProvider(), "http://localhost:4318/v1/logs", {})

    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    assert len(added) == 1
    assert isinstance(added[0], BatchLogRecordProcessor)
    # Shut down the background thread the Batch processor spawned.
    added[0].shutdown()


def test_grpc_log_exporter_path_uses_batch_processor():
    logs = importlib.import_module("cognee.modules.observability.logs")
    if not logs._OTEL_LOGS_AVAILABLE:
        pytest.skip("OTel logs SDK not installed")
    try:
        importlib.import_module("opentelemetry.exporter.otlp.proto.grpc._log_exporter")
    except ImportError:
        pytest.skip("OTLP gRPC exporter not installed")

    added = []

    class FakeProvider:
        def add_log_record_processor(self, processor):
            added.append(processor)

    with patch.dict("os.environ", {}, clear=False):
        logs._try_add_otlp_log_exporter(FakeProvider(), "http://localhost:4317", {})

    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    assert added and all(isinstance(p, BatchLogRecordProcessor) for p in added)
    for processor in added:
        processor.shutdown()
