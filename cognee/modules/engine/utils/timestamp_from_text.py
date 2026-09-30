import re
from datetime import datetime, timezone

from cognee.modules.engine.models import Timestamp
from cognee.modules.engine.models.Timestamp import TimestampPrecision

# The normalized forms the graph prompt asks for, most to least precise. Anchored
# with fullmatch, so "1969-07-20 (Sunday)" or "July 1969" never parse: the LLM was
# told to normalize, and an unparsed name stays an ordinary entity.
_FORMATS: tuple[tuple[re.Pattern, TimestampPrecision], ...] = (
    (re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})"), "second"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "day"),
    (re.compile(r"(\d{4})-(\d{2})"), "month"),
    (re.compile(r"(\d{4})"), "year"),
)
# Earliest value for each unstated part, in field order after the year.
_LOWER_BOUND_DEFAULTS = (1, 1, 0, 0, 0)


def timestamp_from_text(text: str) -> Timestamp | None:
    """The ``Timestamp`` datapoint for a normalized time string, or None.

    Accepts exactly ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` and
    ``YYYY-MM-DD HH:MM:SS``. The precision is the form that matched; unstated
    parts take the earliest value (January, the 1st, midnight), so the
    calendar fields and ``time_at`` are the lower bound of the stated period.
    A string in the right shape that is not a real date (``1950-02-30``,
    ``0000``) returns None as well.
    """
    normalized = text.strip()
    for pattern, precision in _FORMATS:
        match = pattern.fullmatch(normalized)
        if match:
            break
    else:
        return None

    stated = [int(group) for group in match.groups()]
    year, month, day, hour, minute, second = stated + list(_LOWER_BOUND_DEFAULTS[len(stated) - 1 :])
    try:
        lower_bound = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None

    return Timestamp(
        name=normalized,
        timestamp_str=normalized,
        precision=precision,
        time_at=int(lower_bound.timestamp() * 1000),
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )
