"""Timestamp/QueryInterval accept the flat spellings LLMs return (#5375)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cognee.tasks.temporal_graph.models import Event, EventList, QueryInterval, Timestamp


def _fields(ts: Timestamp) -> tuple:
    return (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)


def test_query_interval_accepts_iso_datetime_strings():
    # The exact response from the issue report.
    interval = QueryInterval.model_validate_json(
        '{"starts_at": "2026-10-03T00:00:00", "ends_at": "2026-10-05T13:45:30"}'
    )

    assert _fields(interval.starts_at) == (2026, 10, 3, 0, 0, 0)
    assert _fields(interval.ends_at) == (2026, 10, 5, 13, 45, 30)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-10-03", (2026, 10, 3, 0, 0, 0)),
        ("2026-10-03T08:15:00Z", (2026, 10, 3, 8, 15, 0)),
        ("2026-10-03T23:30:00+05:30", (2026, 10, 3, 23, 30, 0)),
        ("2026-10", (2026, 10, 1, 0, 0, 0)),
        ("2009", (2009, 1, 1, 0, 0, 0)),
        (2009, (2009, 1, 1, 0, 0, 0)),
        (datetime(2020, 2, 29, 12, 0, 1, tzinfo=timezone.utc), (2020, 2, 29, 12, 0, 1)),
    ],
)
def test_timestamp_coerces_flat_values(value, expected):
    assert _fields(Timestamp.model_validate(value)) == expected


def test_timestamp_now_resolves_to_current_utc_date():
    ts = Timestamp.model_validate("now")
    today = datetime.now(timezone.utc)

    assert (ts.year, ts.month, ts.day) == (today.year, today.month, today.day)


def test_nested_object_still_validates():
    interval = QueryInterval.model_validate_json(
        '{"starts_at": {"year": 2010}, "ends_at": {"year": 2020, "month": 3, "day": 5}}'
    )

    assert _fields(interval.starts_at) == (2010, 1, 1, 0, 0, 0)
    assert _fields(interval.ends_at) == (2020, 3, 5, 0, 0, 0)


@pytest.mark.parametrize("empty", [None, "None", "null", ""])
def test_query_interval_treats_empty_spellings_as_none(empty):
    interval = QueryInterval.model_validate({"starts_at": empty, "ends_at": "2009"})

    assert interval.starts_at is None
    assert interval.ends_at.year == 2009


def test_event_list_accepts_iso_strings():
    events = EventList.model_validate_json(
        '{"events": [{"name": "launch", "time_from": "2024-01-15", "time_to": "null"}]}'
    )

    event: Event = events.events[0]
    assert _fields(event.time_from) == (2024, 1, 15, 0, 0, 0)
    assert event.time_to is None


@pytest.mark.parametrize("value", ["last spring", "2026-13-01", True])
def test_unparseable_values_still_fail_validation(value):
    with pytest.raises(ValidationError):
        Timestamp.model_validate(value)
