"""The one accepted shape of a sync policy, validated at the boundary."""

from functools import cache
from zoneinfo import available_timezones

from pydantic import ConfigDict, StrictInt, StrictStr, field_validator, model_validator

from cognee.api.DTO import InDTO

DAY_MINUTES = 1440
WEEK_MINUTES = 10080

# Intervals with one slot a day, at a fixed local time. Every preset of a day or
# more must be listed here: next_due gives any other interval slots through the
# day (a test pins this).
ONE_SLOT_INTERVALS = (DAY_MINUTES, WEEK_MINUTES)

# Offered by a UI and served with the policy. The column stores minutes, so a new
# preset under a day is a code change with no migration.
INTERVAL_PRESETS = (15, 30, 60, 120, 240, 360, 480, *ONE_SLOT_INTERVALS)

# Custom whole hours. Slots under a day restart at local midnight on every listed
# day, so a custom interval of a day or more would have no meaning; "1 day" and
# "1 week" with weekdays cover those.
CUSTOM_HOURS = range(1, 24)

ALL_WEEKDAYS = tuple(range(7))


def _is_allowed_interval(minutes: int) -> bool:
    return minutes in INTERVAL_PRESETS or (minutes % 60 == 0 and minutes // 60 in CUSTOM_HOURS)


@cache
def _timezone_names() -> frozenset[str]:
    # Membership in the tz database, not a file lookup: ZoneInfo reads files, so a
    # long or directory-like name raises OSError and acceptance depends on the OS.
    return frozenset(available_timezones())


class SyncPolicy(InDTO):
    """How often a connection syncs, on which local weekdays, in which timezone."""

    # Frozen, with a tuple for weekdays: a policy is validated once, when it is built, and
    # nothing changes it afterwards. Changing the interval and the weekdays together
    # (weekly takes one day) means building a new one; ``model_copy(update=...)`` skips
    # validation (pydantic), so it is not a way to change one.
    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_minutes: StrictInt
    weekdays: tuple[StrictInt, ...] = ALL_WEEKDAYS
    timezone: StrictStr = "UTC"

    @field_validator("interval_minutes")
    @classmethod
    def valid_interval(cls, value: int) -> int:
        if not _is_allowed_interval(value):
            raise ValueError(
                f"interval_minutes must be one of {list(INTERVAL_PRESETS)} or whole hours "
                f"from {CUSTOM_HOURS.start} to {CUSTOM_HOURS.stop - 1} "
                f"({CUSTOM_HOURS.start * 60} to {(CUSTOM_HOURS.stop - 1) * 60} minutes)"
            )
        return value

    @field_validator("weekdays")
    @classmethod
    def valid_weekdays(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(day not in ALL_WEEKDAYS for day in value):
            raise ValueError("weekdays must contain days from 0 (Monday) to 6 (Sunday)")
        return tuple(sorted(set(value)))

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        if value not in _timezone_names():
            raise ValueError("timezone must be an IANA timezone name, e.g. Europe/Belgrade")
        return value

    @model_validator(mode="after")
    def one_day_for_weekly(self) -> "SyncPolicy":
        if self.interval_minutes == WEEK_MINUTES and len(self.weekdays) != 1:
            raise ValueError("a weekly interval takes exactly one weekday")
        return self
