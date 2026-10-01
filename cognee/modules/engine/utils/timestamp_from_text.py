import re
from datetime import datetime, timedelta, timezone

from cognee.modules.engine.models import Timestamp
from cognee.modules.engine.models.Timestamp import TimestampPrecision
from cognee.modules.engine.utils.temporal_hints import normalize_absolute_date

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

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

# A period with both bounds stated. The normalized form is ``<start>/<end>``
# (ISO 8601 interval notation); prose spans — "from 1803 to 1815",
# "between March 2024 and June 2024", "1803–1815" — are split on the connector
# and each half normalized on its own. A hyphen counts only between two
# four-digit years, so ``1803-05`` stays a month.
_SPAN_SEPARATOR = re.compile(
    r"\s*/\s*|\s+(?:to|and|until|till|through)\s+|\s*[\u2013\u2014]\s*|(?<=\d{4})-(?=\d{4})",
    re.IGNORECASE,
)
_SPAN_PREFIX = re.compile(r"^(?:from|between)\s+", re.IGNORECASE)


def _match_normalized(text: str) -> tuple[re.Match | None, TimestampPrecision | None]:
    for pattern, precision in _FORMATS:
        match = pattern.fullmatch(text)
        if match:
            return match, precision
    return None, None


def _period_end(lower: datetime, precision: TimestampPrecision) -> datetime:
    """The exclusive end of the period ``lower`` starts at the given precision.

    Raises ``ValueError`` for year 9999, whose end would need year 10000.
    """
    if precision == "year":
        return lower.replace(year=lower.year + 1)
    if precision == "month":
        if lower.month == 12:
            return lower.replace(year=lower.year + 1, month=1)
        return lower.replace(month=lower.month + 1)
    if precision == "day":
        return lower + timedelta(days=1)
    return lower + timedelta(seconds=1)


def _span_from_text(text: str) -> Timestamp | None:
    """The span ``Timestamp`` for ``<start>/<end>`` or a prose period, or None.

    Both halves must parse as points and the start must come first; the span
    runs from the start's lower bound to the end's exclusive upper bound, so
    ``1803/1815`` is [1803-01-01, 1816-01-01). The calendar fields are the
    start's.
    """
    parts = _SPAN_SEPARATOR.split(_SPAN_PREFIX.sub("", text.strip()), maxsplit=1)
    if len(parts) != 2:
        return None
    start, end = (_point_from_text(part) for part in parts)
    if start is None or end is None or start.time_at >= end.time_at:
        return None
    fields = {
        name: getattr(start, name) for name in ("year", "month", "day", "hour", "minute", "second")
    }
    return Timestamp(
        timestamp_str=f"{start.timestamp_str}/{end.timestamp_str}",
        precision="span",
        time_at=start.time_at,
        time_until=end.time_until,
        **fields,
    )


def timestamp_from_text(text: str) -> Timestamp | None:
    """The ``Timestamp`` datapoint for a normalized time string, or None.

    Accepts ``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD`` and ``YYYY-MM-DD HH:MM:SS``,
    absolute dates in prose ("23 March 1947", "March 1947"), which
    ``normalize_absolute_date`` brings to one of those shapes at the precision
    they state, and periods with both bounds stated (``1803/1815``,
    "from 1803 to 1815", "between March 2024 and June 2024"), which become one
    span (see ``_span_from_text``). Relative or year-less expressions ("that
    spring", "the 1950s") return None. The precision is the form that matched;
    unstated parts take the earliest value (January, the 1st, midnight), so the
    calendar fields and ``time_at`` are the lower bound of the stated period.
    A string in the right shape that is not a real date (``1950-02-30``,
    ``0000``) returns None too.
    """
    return _point_from_text(text) or _span_from_text(text)


def _point_from_text(text: str) -> Timestamp | None:
    """A single-point ``Timestamp`` (no spans); see ``timestamp_from_text``."""
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
        upper_bound = _period_end(lower_bound, precision)
    except ValueError:
        return None

    return Timestamp(
        name=normalized,
        timestamp_str=normalized,
        precision=precision,
        time_at=int(lower_bound.timestamp() * 1000),
        time_until=int(upper_bound.timestamp() * 1000),
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
    ``timestamp_from_text`` accepts; raises ``ValueError`` for anything else
    (including year 9999, whose exclusive upper bound would need year 10000).
    """
    timestamp = timestamp_from_text(text)
    if timestamp is None:
        raise ValueError(f"Unsupported timestamp: {text!r}")
    # Epoch arithmetic rather than fromtimestamp(): on Windows the latter
    # rejects negative values, i.e. every date before 1970.
    return (
        timestamp.timestamp_str,
        _EPOCH + timedelta(milliseconds=timestamp.time_at),
        _EPOCH + timedelta(milliseconds=timestamp.time_until),
    )
