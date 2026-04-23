"""Event hooks for cognee core operations.

A tiny in-process pub/sub so external observers (audit trails, metrics, custom
loggers) can react to ingest and query lifecycle events without being pinned
as a core dependency. Zero external deps.

Public surface:
    - CogneeEvent: dataclass carrying event_type, timestamp, payload
    - subscribe(listener): register a listener, returns an unsubscribe callable
    - emit(event_type, **payload): fire an event to all listeners

Listeners can be sync or async. Exceptions raised by a listener are logged
and swallowed so a bad listener cannot take down the core operation.
"""

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Union

from cognee.shared.logging_utils import get_logger

logger = get_logger("cognee.events")


# Known event types emitted by cognee core. Listeners may filter on these.
INGEST_BEFORE = "ingest.before"
INGEST_AFTER = "ingest.after"
QUERY_BEFORE = "query.before"
QUERY_AFTER = "query.after"


Listener = Callable[["CogneeEvent"], Union[None, Awaitable[None]]]


@dataclass
class CogneeEvent:
    """A single event fired from a cognee core operation."""

    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = field(default_factory=dict)


_listeners: list[Listener] = []


def subscribe(listener: Listener) -> Callable[[], None]:
    """Register a listener. Returns an unsubscribe function."""
    _listeners.append(listener)

    def unsubscribe() -> None:
        if listener in _listeners:
            _listeners.remove(listener)

    return unsubscribe


def clear_listeners() -> None:
    """Remove all registered listeners. Primarily useful for tests."""
    _listeners.clear()


async def emit(event_type: str, **payload: Any) -> None:
    """Fire an event to all registered listeners.

    If no listeners are registered this is a near-noop (single list check).
    Sync listeners are called directly; async listeners are awaited.
    Listener exceptions are logged and suppressed.
    """
    if not _listeners:
        return

    event = CogneeEvent(event_type=event_type, payload=dict(payload))
    # Snapshot the listener list so a listener that unsubscribes during
    # iteration does not skip siblings.
    for fn in list(_listeners):
        try:
            result = fn(event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning(
                "Cognee event listener raised an exception for %s: %s",
                event_type,
                exc,
            )


__all__ = [
    "CogneeEvent",
    "Listener",
    "subscribe",
    "clear_listeners",
    "emit",
    "INGEST_BEFORE",
    "INGEST_AFTER",
    "QUERY_BEFORE",
    "QUERY_AFTER",
]
