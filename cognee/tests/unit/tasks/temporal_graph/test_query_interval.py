import pytest
from pydantic import ValidationError

from cognee.tasks.temporal_graph.models import QueryInterval, Timestamp


def test_query_interval_nested_objects():
    # (a) nested objects (valid, expected)
    data = {
        "starts_at": {"year": 2026, "month": 10, "day": 3, "hour": 0, "minute": 0, "second": 0},
        "ends_at": {"year": 2026, "month": 10, "day": 4, "hour": 0, "minute": 0, "second": 0},
    }
    interval = QueryInterval.model_validate(data)
    assert interval.starts_at.year == 2026
    assert interval.ends_at.day == 4


def test_query_interval_iso_date_string():
    # (b) ISO date string for the fields
    data = {
        "starts_at": "2026-10-03",
        "ends_at": "2026-10-04",
    }
    interval = QueryInterval.model_validate(data)
    assert interval.starts_at.year == 2026
    assert interval.starts_at.month == 10
    assert interval.starts_at.day == 3
    assert interval.ends_at.day == 4


def test_query_interval_iso_datetime_string():
    # (c) ISO datetime string
    data = {
        "starts_at": "2026-10-03T14:30:00",
    }
    interval = QueryInterval.model_validate(data)
    assert interval.starts_at.year == 2026
    assert interval.starts_at.hour == 14
    assert interval.starts_at.minute == 30


def test_query_interval_null_missing():
    # (d) null / missing fields
    data = {"starts_at": None}
    interval = QueryInterval.model_validate(data)
    assert interval.starts_at is None
    assert interval.ends_at is None

    interval = QueryInterval.model_validate({})
    assert interval.starts_at is None
    assert interval.ends_at is None


def test_query_interval_bare_string():
    # (e) the entire payload as a bare string
    interval = QueryInterval.model_validate("2026-10-03T14:30:00")
    assert interval.starts_at.year == 2026
    assert interval.starts_at.month == 10
    assert interval.starts_at.day == 3
    assert interval.starts_at.hour == 14
    assert interval.starts_at.minute == 30
    assert interval.ends_at is None


def test_query_interval_garbage_input():
    # (f) garbage input
    with pytest.raises(ValueError):
        QueryInterval.model_validate("not-a-date")

    with pytest.raises(ValueError):
        QueryInterval.model_validate({"starts_at": "garbage"})
