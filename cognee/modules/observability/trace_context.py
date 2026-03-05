"""Public tracing API: enable/disable tracing and retrieve traces."""

import os
from typing import Optional

from cognee.modules.observability.tracing import (
    CogneeTrace,
    setup_tracing,
    shutdown_tracing,
    get_exporter,
)

_tracing_enabled: bool = False


def enable_tracing(console_output: bool = False) -> None:
    """Enable OpenTelemetry tracing for Cognee.

    Sets up a TracerProvider with an in-memory CogneeSpanExporter.
    Optionally prints spans to the console when *console_output* is True.
    """
    global _tracing_enabled
    setup_tracing(console_output=console_output)
    _tracing_enabled = True


def disable_tracing() -> None:
    """Disable tracing and shut down the TracerProvider."""
    global _tracing_enabled
    shutdown_tracing()
    _tracing_enabled = False


def is_tracing_enabled() -> bool:
    """Return True when tracing is active.

    Checks the module-level flag, then the ``cognee_tracing_enabled`` config
    field, then falls back to the ``COGNEE_TRACING_ENABLED`` env var directly
    (to support runtime changes, e.g. in tests).  When enabled but not yet
    initialized, lazily calls ``enable_tracing()`` if OpenTelemetry is
    available.
    """
    global _tracing_enabled
    if _tracing_enabled:
        return True

    from cognee.base_config import get_base_config

    config = get_base_config()
    enabled = config.cognee_tracing_enabled or os.environ.get(
        "COGNEE_TRACING_ENABLED", ""
    ).lower() in ("true", "1", "yes")

    if enabled:
        try:
            enable_tracing()
        except ImportError:
            # OpenTelemetry not installed — flag as enabled so callers know
            # tracing is desired, but spans will be no-ops via _NullSpan.
            _tracing_enabled = True
        return True
    return False


def get_last_trace() -> Optional[CogneeTrace]:
    """Return the most recent completed trace from the in-memory buffer."""
    exporter = get_exporter()
    if exporter is None:
        return None
    spans = exporter.get_last_trace_spans()
    if spans is None:
        return None
    return CogneeTrace(spans)


def get_all_traces() -> list[CogneeTrace]:
    """Return all traces currently held in the in-memory buffer."""
    exporter = get_exporter()
    if exporter is None:
        return []
    all_traces = exporter.get_all_traces()
    return [CogneeTrace(spans) for spans in all_traces.values()]


def clear_traces() -> None:
    """Clear the in-memory trace buffer."""
    exporter = get_exporter()
    if exporter is not None:
        exporter.clear()
