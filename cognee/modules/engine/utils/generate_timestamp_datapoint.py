from datetime import datetime, timezone
from cognee.modules.engine.models import Interval, Timestamp, Event
from cognee.modules.engine.utils import generate_node_id


def generate_timestamp_datapoint(ts: Timestamp) -> Timestamp:
    """
    Generates a normalized Timestamp datapoint from a given Timestamp model.

    The function converts the provided timestamp into an integer representation,
    constructs a human-readable string format, and creates a new Timestamp object
    with a unique identifier.

    Args:
        ts (Timestamp): The input Timestamp model containing date and time components.

    Returns:
        Timestamp: A new Timestamp object with a generated ID, integer representation,
                   original components, and formatted string.
    """

    time_at = date_to_int(ts)
    timestamp_str = (
        f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d} {ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
    )
    return Timestamp(
        id=generate_node_id(str(time_at)),
        time_at=time_at,
        year=ts.year,
        month=ts.month,
        day=ts.day,
        hour=ts.hour,
        minute=ts.minute,
        second=ts.second,
        timestamp_str=timestamp_str,
    )


def date_to_int(ts: Timestamp) -> int:
    """
    Converts a Timestamp model into an integer representation in milliseconds since the Unix epoch (UTC).

    Args:
        ts (Timestamp): The input Timestamp model containing year, month, day, hour, minute, and second.

    Returns:
        int: The UTC timestamp in milliseconds since January 1, 1970.
    """
    dt = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=timezone.utc)
    time = int(dt.timestamp() * 1000)
    return time
