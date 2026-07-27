"""OpenTelemetry log bridge for Cognee.

Bridges Python's standard logging to OTel LogRecords so log entries from the
core memory operations appear in any connected OTel backend (Grafana, Dash0,
Langfuse, etc.).

Call setup_log_bridge() once during application startup alongside
setup_tracing() / setup_metrics().
"""

import logging
import os
from typing import Optional

try:
    from opentelemetry._logs import set_logger_provider
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    _OTEL_LOGS_AVAILABLE = True
except ImportError:
    _OTEL_LOGS_AVAILABLE = False

_log_provider: Optional[object] = None
_handler: Optional[object] = None

# Loggers that receive the OTel handler (covers all cognee namespaces)
_COGNEE_LOGGER_NAMES = [
    "cognee",
    "cognee.api",
    "cognee.modules",
    "cognee.tasks",
    "cognee.infrastructure",
]


def setup_log_bridge(console_output: bool = False) -> Optional[object]:
    """Attach an OTel logging handler to all cognee loggers.

    Reads the OTLP endpoint from BaseConfig (same as tracing). Returns the
    LoggerProvider, or None when the OTel logs SDK is unavailable.
    """
    global _log_provider, _handler

    if not _OTEL_LOGS_AVAILABLE:
        return None

    from cognee.base_config import get_base_config
    from cognee.version import get_cognee_version
    from cognee.modules.observability.tracing import _parse_otlp_headers

    config = get_base_config()
    version = get_cognee_version()

    resource = Resource.create(
        {
            "service.name": config.otel_service_name,
            "service.version": version,
            "deployment.environment": os.getenv("ENV", "development"),
        }
    )

    _log_provider = LoggerProvider(resource=resource)

    otlp_endpoint = config.otel_exporter_otlp_endpoint
    if otlp_endpoint:
        headers = _parse_otlp_headers(getattr(config, "otel_exporter_otlp_headers", None))
        _try_add_otlp_log_exporter(_log_provider, otlp_endpoint, headers)

    if console_output:
        from opentelemetry.sdk._logs.export import ConsoleLogExporter

        _log_provider.add_log_record_processor(
            SimpleLogRecordProcessor(ConsoleLogExporter())
        )

    set_logger_provider(_log_provider)

    _handler = LoggingHandler(level=logging.DEBUG, logger_provider=_log_provider)

    for name in _COGNEE_LOGGER_NAMES:
        logger = logging.getLogger(name)
        # Avoid adding duplicate handlers on repeated calls
        if not any(isinstance(h, LoggingHandler) for h in logger.handlers):
            logger.addHandler(_handler)

    return _log_provider


def _try_add_otlp_log_exporter(provider, endpoint: str, headers) -> None:
    logs_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT") or (
        endpoint.replace("/v1/traces", "/v1/logs") if "/v1/traces" in endpoint else endpoint
    )

    if "/api/public/otel" in logs_endpoint:
        _add_http_log_exporter(provider, logs_endpoint, headers)
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

        provider.add_log_record_processor(
            SimpleLogRecordProcessor(OTLPLogExporter(endpoint=logs_endpoint, headers=headers))
        )
    except ImportError:
        _add_http_log_exporter(provider, logs_endpoint, headers)


def _add_http_log_exporter(provider, endpoint: str, headers) -> None:
    try:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter as OTLPHttpLogExporter,
        )

        provider.add_log_record_processor(
            SimpleLogRecordProcessor(OTLPHttpLogExporter(endpoint=endpoint, headers=headers))
        )
    except ImportError:
        import warnings

        warnings.warn(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but no OTLP log exporter is installed. "
            "Install with: pip install cognee[tracing]",
            stacklevel=3,
        )


def shutdown_log_bridge(timeout_ms: int = 30_000) -> None:
    """Detach the OTel handler from all cognee loggers and shut down the provider."""
    global _log_provider, _handler

    if _handler is not None:
        for name in _COGNEE_LOGGER_NAMES:
            logging.getLogger(name).removeHandler(_handler)

    if _log_provider is not None and hasattr(_log_provider, "shutdown"):
        _log_provider.shutdown()

    _log_provider = None
    _handler = None
