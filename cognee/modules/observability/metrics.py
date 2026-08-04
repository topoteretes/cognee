"""OpenTelemetry metrics for Cognee — memory-semconv v0.1.0.

Provides setup_metrics(), get_meter(), and pre-built instruments for the core
memory operations: store, retrieve, process, and delete.

All instruments are created lazily; if the OTel SDK is not installed or metrics
have not been set up, every call is a no-op via _NullInstrument.
"""

import os
from typing import Optional

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource

    _OTEL_METRICS_AVAILABLE = True
except ImportError:
    _OTEL_METRICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# memory-semconv v0.1.0 metric names
# ---------------------------------------------------------------------------

# Latency histograms (unit: ms)
MEMORY_OPERATION_DURATION = "memory.operation.duration"

# Item / data volume counters
MEMORY_ITEMS_STORED = "memory.items.stored"
MEMORY_ITEMS_RETRIEVED = "memory.items.retrieved"
MEMORY_ITEMS_DELETED = "memory.items.deleted"

# Query result distribution
MEMORY_QUERY_RESULT_COUNT = "memory.query.result.count"

# Data size (bytes)
MEMORY_DATA_BYTES_STORED = "memory.data.bytes.stored"

# Vector index counters
MEMORY_VECTOR_SEARCHES = "memory.vector.searches"
MEMORY_GRAPH_EDGES_ADDED = "memory.graph.edges.added"
MEMORY_GRAPH_NODES_ADDED = "memory.graph.nodes.added"

# Error counter
MEMORY_OPERATION_ERRORS = "memory.operation.errors"


# ---------------------------------------------------------------------------
# Null-object pattern — zero overhead when metrics are off
# ---------------------------------------------------------------------------


class _NullInstrument:
    """No-op stand-in for any OTel instrument."""

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


_NULL = _NullInstrument()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_meter: Optional[object] = None
_provider: Optional[object] = None

# Instruments (populated by setup_metrics / _ensure_instruments)
_op_duration: object = _NULL
_items_stored: object = _NULL
_items_retrieved: object = _NULL
_items_deleted: object = _NULL
_query_result_count: object = _NULL
_data_bytes_stored: object = _NULL
_vector_searches: object = _NULL
_graph_edges_added: object = _NULL
_graph_nodes_added: object = _NULL
_op_errors: object = _NULL


def _ensure_instruments() -> None:
    """Create metric instruments against the current meter."""
    global _op_duration, _items_stored, _items_retrieved, _items_deleted
    global _query_result_count, _data_bytes_stored, _vector_searches
    global _graph_edges_added, _graph_nodes_added, _op_errors

    if _meter is None or not _OTEL_METRICS_AVAILABLE:
        return

    _op_duration = _meter.create_histogram(
        name=MEMORY_OPERATION_DURATION,
        description="Duration of memory operations",
        unit="ms",
    )
    _items_stored = _meter.create_counter(
        name=MEMORY_ITEMS_STORED,
        description="Total number of memory items stored",
        unit="{item}",
    )
    _items_retrieved = _meter.create_counter(
        name=MEMORY_ITEMS_RETRIEVED,
        description="Total number of memory items retrieved",
        unit="{item}",
    )
    _items_deleted = _meter.create_counter(
        name=MEMORY_ITEMS_DELETED,
        description="Total number of memory items deleted",
        unit="{item}",
    )
    _query_result_count = _meter.create_histogram(
        name=MEMORY_QUERY_RESULT_COUNT,
        description="Distribution of result counts per query",
        unit="{item}",
    )
    _data_bytes_stored = _meter.create_counter(
        name=MEMORY_DATA_BYTES_STORED,
        description="Total bytes stored into the memory system",
        unit="By",
    )
    _vector_searches = _meter.create_counter(
        name=MEMORY_VECTOR_SEARCHES,
        description="Total number of vector similarity searches executed",
        unit="{search}",
    )
    _graph_edges_added = _meter.create_counter(
        name=MEMORY_GRAPH_EDGES_ADDED,
        description="Total graph edges added to the knowledge graph",
        unit="{edge}",
    )
    _graph_nodes_added = _meter.create_counter(
        name=MEMORY_GRAPH_NODES_ADDED,
        description="Total graph nodes added to the knowledge graph",
        unit="{node}",
    )
    _op_errors = _meter.create_counter(
        name=MEMORY_OPERATION_ERRORS,
        description="Total number of memory operation errors",
        unit="{error}",
    )


def setup_metrics(console_output: bool = False) -> object:
    """Configure OTel MeterProvider for Cognee.

    If an external MeterProvider is already configured (e.g. via
    opentelemetry-instrument), it is reused. Otherwise a fresh one is created
    and an OTLP metrics exporter is attached when
    OTEL_EXPORTER_OTLP_ENDPOINT is configured.

    Returns the cognee meter.
    """
    global _meter, _provider

    if not _OTEL_METRICS_AVAILABLE:
        return None

    from cognee.version import get_cognee_version

    version = get_cognee_version()

    current = metrics.get_meter_provider()
    # ProxyMeterProvider is the default before any SDK provider is set
    external = current is not None and type(current).__name__ not in (
        "ProxyMeterProvider",
        "NoOpMeterProvider",
    )

    if external:
        _provider = current
    else:
        from cognee.base_config import get_base_config

        config = get_base_config()
        resource = Resource.create(
            {
                "service.name": config.otel_service_name,
                "service.version": version,
                "deployment.environment": os.getenv("ENV", "development"),
            }
        )

        readers = []

        if console_output:
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(), export_interval_millis=60_000
                )
            )

        otlp_endpoint = config.otel_exporter_otlp_endpoint
        if otlp_endpoint:
            _try_add_otlp_metric_reader(readers, otlp_endpoint, config)

        _provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(_provider)

    _meter = _provider.get_meter("cognee", version)
    _ensure_instruments()
    return _meter


def _try_add_otlp_metric_reader(readers: list, endpoint: str, config) -> None:
    """Try to add an OTLP metric reader; silently skip if package missing."""
    import logging as _logging

    _log = _logging.getLogger("cognee.observability")

    from cognee.modules.observability.tracing import _parse_otlp_headers, _requires_http_exporter

    headers = _parse_otlp_headers(getattr(config, "otel_exporter_otlp_headers", None))

    # Resolve metrics endpoint: standard env var takes precedence, then derive from traces ep
    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or (
        endpoint.replace("/v1/traces", "/v1/metrics") if "/v1/traces" in endpoint else endpoint
    )

    if _requires_http_exporter(metrics_endpoint):
        _add_http_metric_reader(readers, metrics_endpoint, headers)
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=metrics_endpoint, headers=headers),
                export_interval_millis=30_000,
            )
        )
        _log.info("OTel: OTLP gRPC metric exporter registered → %s", metrics_endpoint)
    except ImportError:
        _add_http_metric_reader(readers, metrics_endpoint, headers)


def _add_http_metric_reader(readers: list, endpoint: str, headers) -> None:
    import logging as _logging

    _log = _logging.getLogger("cognee.observability")
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as OTLPHttpMetricExporter,
        )

        readers.append(
            PeriodicExportingMetricReader(
                OTLPHttpMetricExporter(endpoint=endpoint, headers=headers),
                export_interval_millis=30_000,
            )
        )
        _log.info("OTel: OTLP HTTP metric exporter registered → %s", endpoint)
    except ImportError:
        import warnings

        msg = (
            f"OTEL_EXPORTER_OTLP_ENDPOINT is set to '{endpoint}' but no OTLP metrics "
            "exporter is installed. Metrics will NOT be exported. "
            "Install with: pip install cognee[tracing]"
        )
        _log.error("OTel: %s", msg)
        warnings.warn(
            msg,
            stacklevel=3,
        )


def get_meter() -> Optional[object]:
    """Return the cognee OTel meter, or None if not configured."""
    return _meter


# ---------------------------------------------------------------------------
# Public metric helpers (no-op when not configured)
# ---------------------------------------------------------------------------


def record_operation_duration(duration_ms: float, attributes: dict) -> None:
    _op_duration.record(duration_ms, attributes)


def increment_items_stored(count: int = 1, attributes: dict = None) -> None:
    _items_stored.add(count, attributes or {})


def increment_items_retrieved(count: int = 1, attributes: dict = None) -> None:
    _items_retrieved.add(count, attributes or {})


def increment_items_deleted(count: int = 1, attributes: dict = None) -> None:
    _items_deleted.add(count, attributes or {})


def record_query_results(count: int, attributes: dict = None) -> None:
    _query_result_count.record(count, attributes or {})


def increment_bytes_stored(byte_count: int, attributes: dict = None) -> None:
    _data_bytes_stored.add(byte_count, attributes or {})


def increment_vector_searches(attributes: dict = None) -> None:
    _vector_searches.add(1, attributes or {})


def increment_graph_edges(count: int = 1, attributes: dict = None) -> None:
    _graph_edges_added.add(count, attributes or {})


def increment_graph_nodes(count: int = 1, attributes: dict = None) -> None:
    _graph_nodes_added.add(count, attributes or {})


def increment_operation_errors(attributes: dict = None) -> None:
    _op_errors.add(1, attributes or {})


def shutdown_metrics(timeout_ms: int = 30_000) -> None:
    """Shut down the MeterProvider and clear global state."""
    global _meter, _provider

    if _provider is not None and hasattr(_provider, "shutdown"):
        _provider.shutdown()
    _meter = None
    _provider = None
