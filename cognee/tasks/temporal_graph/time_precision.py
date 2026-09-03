"""Precision handling for LLM-extracted timestamps.

The extraction model (``cognee.tasks.temporal_graph.models.Timestamp``) defaults
month and day to 1 and the clock to 0, so a bare "1996" arrives as
1996-01-01 00:00:00. Anything that uses such a value as an *upper* bound — a query
window ending "in 1996", an interval that "lasted until 1996" — must widen it to
the last second of the unit the text actually stated, or everything later in that
year is silently excluded.
"""

import calendar
from typing import Optional

from cognee.tasks.temporal_graph.models import Timestamp

PRECISION_ORDER = ("year", "month", "day", "hour", "minute", "second")


def infer_precision(ts: Timestamp) -> str:
    """Return the explicit ``precision`` or infer it from the default-valued fields.

    Without an explicit ``precision`` the only signal is which fields still hold
    their defaults; a genuine "January 1st" is then treated as a year, which widens
    a bound rather than narrowing it.
    """
    if ts.precision:
        return ts.precision
    if ts.second:
        return "second"
    if ts.minute:
        return "minute"
    if ts.hour:
        return "hour"
    if ts.day != 1:
        return "day"
    if ts.month != 1:
        return "month"
    return "year"


def expand_to_period_end(ts: Optional[Timestamp]) -> Optional[Timestamp]:
    """Widen a timestamp to the last second of the unit it was stated in.

    "1996" becomes 1996-12-31 23:59:59, "March 1996" becomes 1996-03-31 23:59:59,
    "5 March 1996" becomes 1996-03-05 23:59:59. Values stated to the second are
    returned unchanged (apart from carrying their inferred ``precision``).
    """
    if ts is None:
        return None

    precision = infer_precision(ts)
    values = ts.model_dump()
    values["precision"] = precision

    if precision == "year":
        values.update(month=12, day=31, hour=23, minute=59, second=59)
    elif precision == "month":
        values.update(day=calendar.monthrange(ts.year, ts.month)[1], hour=23, minute=59, second=59)
    elif precision == "day":
        values.update(hour=23, minute=59, second=59)
    elif precision == "hour":
        values.update(minute=59, second=59)
    elif precision == "minute":
        values.update(second=59)

    return Timestamp(**values)
