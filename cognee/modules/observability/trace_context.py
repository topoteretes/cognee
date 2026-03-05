"""Public tracing API: enable/disable tracing and retrieve traces."""

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

    Checks both the module-level flag and the ``cognee_tracing_enabled``
    config field.  When the config enables tracing but it has not been
    initialized yet, lazily calls ``enable_tracing()`` so that the
    TracerProvider and exporters are set up.
    """
    global _tracing_enabled
    if _tracing_enabled:
        return True

    from cognee.base_config import get_base_config

    if get_base_config().cognee_tracing_enabled:
        enable_tracing()
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
