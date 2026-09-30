import re
from datetime import datetime, timedelta, timezone

from cognee.modules.engine.models import Timestamp
from cognee.modules.engine.models.Timestamp import TimestampPrecision
from cognee.modules.engine.utils.temporal_hints import normalize_absolute_date

# The normalized forms the graph prompt asks for, most to least precise. Anchored
# with fullmatch, so "1969-07-20 (Sunday)" never parses; a date left in prose
# ("July 1969") gets one normalization pass first, and anything still unparsed
# stays an ordinary entity.
_FORMATS: tuple[tuple[re.Pattern, TimestampPrecision], ...] = (
    (re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})"), "second"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "day"),
    (re.compile(r"(\d{4})-(\d{2})"), "month"),
    (re.compile(r"(\d{4})"), "year"),
)
# Earliest value for each unstated part, in field order after the year.
_LOWER_BOUND_DEFAULTS = (1, 1, 0, 0, 0)


def _match_normalized(text: str) -> tuple[re.Match | None, TimestampPrecision | None]:
    for pattern, precision in _FORMATS:
        match = pattern.fullmatch(text)
        if match:
            return match, precision
    return None, None


def timestamp_from_text(text: str) -> Timestamp | None:
    """The ``Timestamp`` datapoint for a normalized time string, or None.

    Accepts ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` and ``YYYY-MM-DD HH:MM:SS``,
    and absolute dates in prose ("23 March 1947", "March 1947"), which
    ``normalize_absolute_date`` brings to one of those shapes at the precision
    they state. Relative or year-less expressions ("that spring", "the 1950s")
    return None. The precision is the form that matched; unstated parts take
    the earliest value (January, the 1st, midnight), so the calendar fields and
    ``time_at`` are the lower bound of the stated period. A string in the right
    shape that is not a real date (``1950-02-30``, ``0000``) returns None too.
    """
    match, precision = _match_normalized(text.strip())
    if match is None:
        normalized = normalize_absolute_date(text)
        if normalized is None:
            return None
        match, precision = _match_normalized(normalized)
        if match is None:
            return None
    normalized = match.group(0)

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


def timestamp_bounds(text: str) -> tuple[str, datetime, datetime]:
    """``(normalized, lower, upper)`` for a time string: the half-open period it names.

    ``1950`` is [1950-01-01, 1951-01-01), ``1950-03`` is [1950-03-01, 1950-04-01),
    a day is one day, a full timestamp is one second. Accepts what
    ``timestamp_from_text`` accepts; raises ``ValueError`` for anything else, and
    for year 9999 (the exclusive upper bound would need year 10000).
    """
    timestamp = timestamp_from_text(text)
    if timestamp is None:
        raise ValueError(f"Unsupported timestamp: {text!r}")
    if timestamp.year > 9998:
        raise ValueError(f"Unsupported timestamp: {text!r}")
    lower = datetime(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        tzinfo=timezone.utc,
    )
    if timestamp.precision == "year":
        upper = lower.replace(year=lower.year + 1)
    elif timestamp.precision == "month":
        upper = (
            lower.replace(year=lower.year + 1, month=1)
            if lower.month == 12
            else lower.replace(month=lower.month + 1)
        )
    elif timestamp.precision == "day":
        upper = lower + timedelta(days=1)
    else:
        upper = lower + timedelta(seconds=1)
    return timestamp.timestamp_str, lower, upper
