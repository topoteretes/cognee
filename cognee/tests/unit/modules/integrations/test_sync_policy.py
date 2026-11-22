"""SyncPolicy validation and next_due slots (SDK-1434).

The DST cases mirror the ones SDK-1318 pinned for ``next_sync_at``, whose slot
walk ``next_due`` carries over.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from cognee.modules.integrations.sync_policy import INTERVAL_PRESETS, SyncPolicy, next_due
from cognee.modules.integrations.sync_policy.policy import DAY_MINUTES, ONE_SLOT_INTERVALS

UTC = timezone.utc


def _utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=UTC)


@pytest.mark.parametrize("minutes", [*INTERVAL_PRESETS, 60, 300, 1380])
def test_presets_and_custom_whole_hours_are_accepted(minutes):
    weekdays = [0] if minutes == 10080 else [0, 1, 2, 3, 4, 5, 6]
    assert SyncPolicy(interval_minutes=minutes, weekdays=weekdays).interval_minutes == minutes


@pytest.mark.parametrize(
    "payload",
    [
        {"interval_minutes": 0},
        {"interval_minutes": 45},
        {"interval_minutes": 90},
        {"interval_minutes": 1500},
        {"interval_minutes": True},
        {"interval_minutes": "60"},
        {"interval_minutes": 60, "weekdays": []},
        {"interval_minutes": 60, "weekdays": [7]},
        {"interval_minutes": 60, "weekdays": [-1]},
        {"interval_minutes": 60, "weekdays": ["1"]},
        {"interval_minutes": 60, "weekdays": [True]},
        {"interval_minutes": 60, "weekdays": [1.0]},
        {"interval_minutes": 60, "timezone": "Missing/Zone"},
        # Names ZoneInfo would try as files: a directory, a path, a NUL, a long name.
        {"interval_minutes": 60, "timezone": "America"},
        {"interval_minutes": 60, "timezone": "../etc"},
        {"interval_minutes": 60, "timezone": "\x00"},
        {"interval_minutes": 60, "timezone": "x" * 5000},
        # Not canonical keys: some filesystems resolve them, so acceptance would vary.
        {"interval_minutes": 60, "timezone": ""},
        {"interval_minutes": 60, "timezone": "UTC "},
        {"interval_minutes": 60, "timezone": "utc"},
        {"interval_minutes": 60, "timezone": b"UTC"},
        {"interval_minutes": 60, "timezone": None},
        {"interval_minutes": 10080, "weekdays": [0, 1]},
        {"interval_minutes": 60, "window": "09:00-17:00"},
    ],
)
def test_anything_else_is_refused(payload):
    with pytest.raises(ValidationError):
        SyncPolicy(**payload)


def test_refusal_names_the_presets_and_the_custom_range():
    with pytest.raises(ValidationError) as error:
        SyncPolicy(interval_minutes=45)

    assert "[15, 30, 60, 120, 240, 360, 480, 1440, 10080]" in str(error.value)
    assert "60 to 1380 minutes" in str(error.value)


def test_an_unknown_timezone_is_a_validation_error_that_names_the_remedy():
    with pytest.raises(ValidationError) as error:
        SyncPolicy(interval_minutes=60, timezone="")

    assert "IANA timezone name" in str(error.value)


def test_a_policy_is_immutable_so_it_is_validated_only_when_built():
    policy = SyncPolicy(interval_minutes=60)

    with pytest.raises(ValidationError):
        policy.weekdays = []
    with pytest.raises(ValidationError):
        policy.interval_minutes = 10080
    with pytest.raises(AttributeError):
        policy.weekdays.clear()
    assert policy.weekdays == (0, 1, 2, 3, 4, 5, 6)
    assert policy.interval_minutes == 60
    assert hash(policy) == hash(SyncPolicy(interval_minutes=60))


def test_every_preset_of_a_day_or_more_has_one_slot_a_day():
    """next_due gives any other interval slots through the day, so a new long preset must be listed."""
    assert [m for m in INTERVAL_PRESETS if m >= DAY_MINUTES] == list(ONE_SLOT_INTERVALS)


def test_camel_case_body_and_weekdays_normalised():
    policy = SyncPolicy.model_validate({"intervalMinutes": 60, "weekdays": [3, 1, 3]})

    assert policy.weekdays == (1, 3)
    assert policy.model_dump(by_alias=True, mode="json")["weekdays"] == [1, 3]


def test_four_hours_from_local_midnight():
    policy = SyncPolicy(interval_minutes=240)

    assert next_due(policy, _utc(2026, 11, 5, 0, 0)) == _utc(2026, 11, 5, 4, 0)
    assert next_due(policy, _utc(2026, 11, 5, 21, 0)) == _utc(2026, 11, 6, 0, 0)


def test_custom_hours_restart_at_the_next_midnight():
    policy = SyncPolicy(interval_minutes=300)

    assert next_due(policy, _utc(2026, 11, 5, 15, 0)) == _utc(2026, 11, 5, 20, 0)
    # 20:00 is the day's last slot; the next one is midnight, not 01:00.
    assert next_due(policy, _utc(2026, 11, 5, 20, 0)) == _utc(2026, 11, 6, 0, 0)


def test_daily_runs_at_ten_local_on_listed_days_only():
    policy = SyncPolicy(interval_minutes=1440, weekdays=[0, 1, 2, 3, 4], timezone="Europe/Belgrade")

    # Thursday 15:00 local: next is Friday 10:00 local (09:00 UTC in winter).
    assert next_due(policy, _utc(2026, 11, 5, 14, 0)) == _utc(2026, 11, 6, 9, 0)
    # Friday after the slot: the weekend is skipped.
    assert next_due(policy, _utc(2026, 11, 6, 9, 0)) == _utc(2026, 11, 9, 9, 0)


def test_daily_keeps_ten_local_across_the_spring_change():
    policy = SyncPolicy(interval_minutes=1440, timezone="Europe/Belgrade")

    # 28 Mar is CET (UTC+1), 29 Mar is CEST (UTC+2).
    assert next_due(policy, _utc(2026, 3, 28, 9, 0)) == _utc(2026, 3, 29, 8, 0)


def test_weekly_runs_on_its_one_weekday():
    policy = SyncPolicy(interval_minutes=10080, weekdays=[0])

    assert next_due(policy, _utc(2026, 11, 5, 12, 0)) == _utc(2026, 11, 9, 10, 0)


def test_spring_gap_slots_are_skipped():
    policy = SyncPolicy(interval_minutes=30, timezone="Europe/Berlin")

    # 01:30 local on 29 Mar 2026; 02:00 and 02:30 do not exist, 03:00 CEST does.
    assert next_due(policy, _utc(2026, 3, 29, 0, 30)) == _utc(2026, 3, 29, 1, 0)


def test_a_listed_day_whose_midnight_does_not_exist_starts_at_its_first_real_slot():
    # Santiago springs forward at local midnight: Sunday 6 Sep 2026 has no 00:00.
    policy = SyncPolicy(interval_minutes=60, weekdays=[6], timezone="America/Santiago")

    slot = next_due(policy, _utc(2026, 9, 6, 2, 30))  # Saturday 22:30 local

    assert slot.astimezone(ZoneInfo("America/Santiago")).strftime("%a %H:%M") == "Sun 01:00"


def test_the_earlier_slot_wins_when_the_repeated_hour_is_listed_first():
    policy = SyncPolicy(interval_minutes=15, timezone="Europe/Berlin")

    # 00:01 UTC is 02:01 CEST on the autumn change. 02:00 CET (01:00 UTC) is also a candidate
    # that day, and it must not beat 02:15 CEST (00:15 UTC).
    assert next_due(policy, _utc(2026, 10, 25, 0, 1)) == _utc(2026, 10, 25, 0, 15)


def test_the_local_date_not_the_utc_date_picks_the_day():
    policy = SyncPolicy(interval_minutes=60, timezone="America/New_York")

    # 02:00 UTC on 5 Nov is still 4 Nov, 21:00 local: the next slot is 22:00 local.
    assert next_due(policy, _utc(2026, 11, 5, 2, 0)) == _utc(2026, 11, 5, 3, 0)


def test_the_last_slot_of_a_day_is_not_midnight_of_the_next_unlisted_day():
    policy = SyncPolicy(interval_minutes=60, weekdays=[0])  # Mondays only

    # Monday 23:30: the rest of Monday has no slot, and Tuesday 00:00 is not a listed day.
    assert next_due(policy, _utc(2026, 11, 2, 23, 30)) == _utc(2026, 11, 9, 0, 0)


def test_autumn_fold_runs_the_repeated_hour_twice():
    policy = SyncPolicy(interval_minutes=60, timezone="Europe/Berlin")

    # 02:00 local happens at 00:00 UTC (CEST) and again at 01:00 UTC (CET).
    assert next_due(policy, _utc(2026, 10, 24, 23, 30)) == _utc(2026, 10, 25, 0, 0)
    assert next_due(policy, _utc(2026, 10, 25, 0, 0)) == _utc(2026, 10, 25, 1, 0)
    assert next_due(policy, _utc(2026, 10, 25, 1, 0)) == _utc(2026, 10, 25, 2, 0)


def test_naive_clock_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        next_due(SyncPolicy(interval_minutes=60), _utc(2026, 11, 5, 12, 0).replace(tzinfo=None))
