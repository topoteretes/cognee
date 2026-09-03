"""One reading of "is this fact still current?" for retrieval.

Cognee marks a fact as no longer current in two ways that used to be invisible to
search: ``close_node`` stamps ``valid_to`` (ms epoch) on a node, and
``resolve_temporal_contradictions`` tags the edge a newer assertion replaced with
``superseded`` (and, since SDK-90, the same ``valid_to``). Retrieval reads both
through the helpers here, so a stale fact is ranked below a current one and is
rendered with an explicit "until" marker instead of passing as present truth.
"""

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

# A stale fact is not dropped — "who was CEO in 2019" needs it — but its triplet
# distance is scaled up so an equally relevant current fact outranks it.
STALE_DISTANCE_FACTOR = 1.5

_EDGE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def to_epoch_ms(value: Any) -> Optional[int]:
    """Coerce a stored timestamp to ms epoch.

    Accepts the ms-epoch ints ``DataPoint`` uses, numeric strings, and the
    ``"%Y-%m-%d %H:%M:%S"`` UTC strings ``get_graph_from_model`` stamps on edges.
    Returns ``None`` for anything unparseable.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            pass
        try:
            parsed = datetime.strptime(value, _EDGE_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            return None
    return None


def is_current(attributes: Optional[Mapping[str, Any]], as_of_ms: Optional[int] = None) -> bool:
    """True unless the element is superseded or its ``valid_to`` is at or before ``as_of_ms``.

    ``as_of_ms`` defaults to now. A fact closed *after* the reference time is still
    current for that time, which is what a query about the past needs.
    """
    if not attributes:
        return True
    if attributes.get("superseded"):
        return False
    valid_to = to_epoch_ms(attributes.get("valid_to"))
    if valid_to is None:
        return True
    reference = now_ms() if as_of_ms is None else as_of_ms
    return valid_to > reference


def validity_marker(attributes: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Short human-readable note for a fact that is no longer current, else ``None``.

    Independent of any reference time: the marker says what the graph records, the
    ranking decides how much it matters for the question at hand.
    """
    if not attributes:
        return None
    valid_to = to_epoch_ms(attributes.get("valid_to"))
    superseded = bool(attributes.get("superseded"))
    if valid_to is None and not superseded:
        return None
    parts = []
    if valid_to is not None:
        until = datetime.fromtimestamp(valid_to / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        parts.append(f"valid until {until}")
    if superseded:
        parts.append("superseded by a newer assertion")
    return "; ".join(parts)
