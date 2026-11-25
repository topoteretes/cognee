"""The next slot of a sync policy, in UTC.

The walk goes through local wall times and converts each one, so a slot that
does not exist on a spring-forward day is skipped and a slot inside the repeated
autumn hour runs in both folds.
"""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .policy import ONE_SLOT_INTERVALS, SyncPolicy

# The single local time of day for daily and weekly policies.
DAILY_SLOT_TIME = time(10, 0)


def next_due(policy: SyncPolicy, after: datetime) -> datetime:
    """The first slot strictly after ``after``.

    Under a day, slots start at local midnight on every listed weekday and repeat
    every ``interval_minutes`` until midnight, then start again on the next
    listed day. A day or a week has one slot, at 10:00 local on each listed day.
    """
    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")
    zone = ZoneInfo(policy.timezone)
    start = DAILY_SLOT_TIME if policy.interval_minutes in ONE_SLOT_INTERVALS else time()
    step = timedelta(minutes=policy.interval_minutes)
    day = after.astimezone(zone).date()
    # Validation guarantees at least one weekday, so a slot comes within a week.
    while True:
        if day.weekday() in policy.weekdays:
            midnight = datetime.combine(day + timedelta(days=1), time())
            wall = datetime.combine(day, start)
            candidates = []
            while wall < midnight:
                for fold in (0, 1):
                    candidate = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
                    # A wall time that does not round-trip does not exist on this day.
                    if candidate.astimezone(zone).replace(tzinfo=None) != wall:
                        continue
                    if candidate > after:
                        candidates.append(candidate)
                wall += step
            if candidates:
                return min(candidates)
        day += timedelta(days=1)
