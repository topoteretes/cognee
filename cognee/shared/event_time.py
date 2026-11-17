"""Event times as the session cache stores them.

Kept free of cognee imports: ``cognee.memory.entries`` validates with it at import
time, before the cache package (whose engine import reaches the graph adapters) loads.
"""

from datetime import datetime, timezone


def normalize_event_time(value: str) -> str:
    """An ISO-8601 event time as the cache stores it: timezone-aware, in UTC.

    The client's clock stamps the event (the plugin's Stop hook for an answer, its
    PostToolUse hook for a trace), so a session's entries compare on one clock
    whatever order they arrived in. A naive or unparsable value is refused.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("time must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"time is not an ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError("time must carry a timezone offset")
    return parsed.astimezone(timezone.utc).isoformat()
