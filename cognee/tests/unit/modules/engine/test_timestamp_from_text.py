from datetime import datetime, timezone

import pytest

from cognee.modules.engine.models import Timestamp
from cognee.modules.engine.utils import generate_timestamp_datapoint, timestamp_from_text
from cognee.tasks.temporal_graph.models import Timestamp as LLMTimestamp


def _epoch_ms(*parts) -> int:
    return int(datetime(*parts, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.parametrize(
    "text, precision, fields, lower_bound",
    [
        ("1969", "year", (1969, 1, 1, 0, 0, 0), _epoch_ms(1969, 1, 1)),
        ("1969-07", "month", (1969, 7, 1, 0, 0, 0), _epoch_ms(1969, 7, 1)),
        ("1969-07-20", "day", (1969, 7, 20, 0, 0, 0), _epoch_ms(1969, 7, 20)),
        (
            "1969-07-20 20:17:40",
            "second",
            (1969, 7, 20, 20, 17, 40),
            _epoch_ms(1969, 7, 20, 20, 17, 40),
        ),
    ],
)
def test_four_precisions_fill_unstated_parts_with_the_lower_bound(
    text, precision, fields, lower_bound
):
    timestamp = timestamp_from_text(text)

    assert isinstance(timestamp, Timestamp)
    assert timestamp.precision == precision
    assert timestamp.timestamp_str == text
    assert timestamp.name == text
    fields_on_timestamp = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
    assert fields_on_timestamp == fields
    assert timestamp.time_at == lower_bound


def test_surrounding_whitespace_is_ignored():
    assert timestamp_from_text("  2024-02-29 ").timestamp_str == "2024-02-29"


@pytest.mark.parametrize(
    "text",
    [
        "1940s",
        "",
        "Apollo 11",
    ],
)
def test_anything_but_the_normalized_forms_is_rejected(text):
    assert timestamp_from_text(text) is None


@pytest.mark.parametrize(
    "text", ["0000", "1950-13", "1950-02-30", "2023-02-29", "1950-01-01 24:00:00"]
)
def test_a_well_shaped_string_that_is_not_a_real_time_is_rejected(text):
    assert timestamp_from_text(text) is None


def test_id_derives_from_the_normalized_string():
    assert timestamp_from_text("1969").id == timestamp_from_text(" 1969 ").id
    assert timestamp_from_text("1969").id == Timestamp.id_for("1969")
    # Same calendar fields, different stated precision: two nodes.
    assert timestamp_from_text("1969").id != timestamp_from_text("1969-01-01").id


def test_temporal_pipeline_timestamps_keep_their_explicit_id_and_gain_a_name():
    timestamp = generate_timestamp_datapoint(LLMTimestamp(year=1969, month=7, day=20))

    assert timestamp.timestamp_str == "1969-07-20 00:00:00"
    assert timestamp.name == timestamp.timestamp_str
    assert timestamp.precision == "second"
    assert timestamp.id != Timestamp.id_for(timestamp.timestamp_str)


@pytest.mark.parametrize(
    "text, normalized, precision",
    [
        ("23 March 1947", "1947-03-23", "day"),
        ("March 1947", "1947-03", "month"),
        ("April 27, 1791", "1791-04-27", "day"),
        ("1969-07-20 (Sunday)", "1969-07-20", "day"),
        ("20/07/1969", "1969-07-20", "day"),
        ("1969-7-20", "1969-07-20", "day"),
        ("1969-07-20 20:17", "1969-07-20 20:17:00", "second"),
    ],
)
def test_an_absolute_date_in_prose_is_normalized_first(text, normalized, precision):
    timestamp = timestamp_from_text(text)

    assert timestamp.timestamp_str == normalized
    assert timestamp.name == normalized
    assert timestamp.precision == precision
    assert timestamp.id == Timestamp.id_for(normalized)


@pytest.mark.parametrize(
    "text", ["the 1950s", "that spring", "four weeks later", "spring of 1943", "July 24", "24 July"]
)
def test_relative_or_year_less_prose_is_rejected(text):
    assert timestamp_from_text(text) is None


def test_a_timestamp_built_without_a_name_is_labelled_by_its_string():
    """Existing callers (temporal pipeline, retriever tests) construct Timestamps without ``name``."""
    timestamp = Timestamp(
        time_at=1609459200000,
        year=2021,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        timestamp_str="2021-01-01T00:00:00",
    )

    assert timestamp.name == "2021-01-01T00:00:00"
    assert timestamp.id == Timestamp.id_for("2021-01-01T00:00:00")


@pytest.mark.parametrize(
    "text, normalized, lower, upper",
    [
        ("1950", "1950", (1950, 1, 1), (1951, 1, 1)),
        ("1950-12", "1950-12", (1950, 12, 1), (1951, 1, 1)),
        ("2024-02-29", "2024-02-29", (2024, 2, 29), (2024, 3, 1)),
        (
            "1969-07-20 20:17:40",
            "1969-07-20 20:17:40",
            (1969, 7, 20, 20, 17, 40),
            (1969, 7, 20, 20, 17, 41),
        ),
        ("March 1947", "1947-03", (1947, 3, 1), (1947, 4, 1)),
    ],
)
def test_timestamp_bounds_is_the_half_open_period_at_the_stated_precision(
    text, normalized, lower, upper
):
    from cognee.modules.engine.utils.timestamp_from_text import timestamp_bounds

    got_normalized, got_lower, got_upper = timestamp_bounds(text)

    assert got_normalized == normalized
    assert got_lower == datetime(*lower, tzinfo=timezone.utc)
    assert got_upper == datetime(*upper, tzinfo=timezone.utc)


@pytest.mark.parametrize("text", ["1940s", "that spring", "9999", "1950-02-30"])
def test_timestamp_bounds_rejects_what_the_parser_rejects_and_year_9999(text):
    from cognee.modules.engine.utils.timestamp_from_text import timestamp_bounds

    with pytest.raises(ValueError):
        timestamp_bounds(text)


# --- spans: a period with both bounds stated (SDK-899) -------------------------


@pytest.mark.parametrize(
    "text, normalized, lower, upper",
    [
        ("1803/1815", "1803/1815", (1803, 1, 1), (1816, 1, 1)),
        ("from 1803 to 1815", "1803/1815", (1803, 1, 1), (1816, 1, 1)),
        ("between 1803 and 1815", "1803/1815", (1803, 1, 1), (1816, 1, 1)),
        ("1803–1815", "1803/1815", (1803, 1, 1), (1816, 1, 1)),
        ("1803-1815", "1803/1815", (1803, 1, 1), (1816, 1, 1)),
        ("2024-03/2024-06", "2024-03/2024-06", (2024, 3, 1), (2024, 7, 1)),
        ("between March 2024 and June 2024", "2024-03/2024-06", (2024, 3, 1), (2024, 7, 1)),
        ("from 12 May 2023 to 14 May 2023", "2023-05-12/2023-05-14", (2023, 5, 12), (2023, 5, 15)),
        ("1969-07-16 / 1969-07-24", "1969-07-16/1969-07-24", (1969, 7, 16), (1969, 7, 25)),
    ],
)
def test_a_period_with_both_bounds_is_one_span_timestamp(text, normalized, lower, upper):
    timestamp = timestamp_from_text(text)

    assert timestamp is not None
    assert timestamp.precision == "span"
    assert timestamp.timestamp_str == normalized
    assert timestamp.name == normalized
    assert timestamp.time_at == _epoch_ms(*lower)
    assert timestamp.time_until == _epoch_ms(*upper)
    assert (timestamp.year, timestamp.month, timestamp.day) == lower
    assert timestamp.id == Timestamp.id_for(normalized)


@pytest.mark.parametrize(
    "text",
    [
        "1815/1803",  # backwards
        "1803/1803",  # not a period
        "from 1803 to that spring",  # one half does not parse
        "between March and June 2024",  # first half has no year
        "1803/1815/1820",  # three bounds
    ],
)
def test_malformed_periods_are_rejected(text):
    assert timestamp_from_text(text) is None


def test_a_hyphen_splits_only_two_four_digit_years():
    assert timestamp_from_text("1803-05").precision == "month"
    assert timestamp_from_text("1803-1815").precision == "span"


def test_a_window_inside_a_span_overlaps_it():
    from cognee.infrastructure.databases.graph.graph_db_interface import timestamp_overlaps

    span = timestamp_from_text("from 1803 to 1815")
    node = {"id": "ts", "time_at": span.time_at, "time_until": span.time_until}
    assert timestamp_overlaps(node, _epoch_ms(1805, 1, 1), _epoch_ms(1806, 1, 1)) is True
    assert timestamp_overlaps(node, _epoch_ms(1816, 1, 1), _epoch_ms(1817, 1, 1)) is False
    assert timestamp_overlaps(node, _epoch_ms(1802, 1, 1), _epoch_ms(1803, 1, 1)) is False


@pytest.mark.parametrize(
    "text",
    ["1969", "1969-12", "1969-12-31", "1968/1969", "1969-12-31 23:59:59"],
)
def test_a_period_ending_at_the_epoch_keeps_its_real_upper_bound(text):
    """``time_until`` of 0 is 1970-01-01T00:00:00Z, not "missing": the default of
    one second after ``time_at`` must apply only when no bound was given."""
    timestamp = timestamp_from_text(text)

    assert timestamp.time_until == 0
    # A July-1969 window overlaps a date that spans all of 1969.
    if text == "1969":
        assert timestamp.time_at < _epoch_ms(1969, 8, 1) and timestamp.time_until > _epoch_ms(
            1969, 7, 1
        )


def test_time_until_defaults_only_when_absent():
    base = {"timestamp_str": "1970-01-01 00:00:00", "time_at": 0, "year": 1970, "month": 1}
    base.update(day=1, hour=0, minute=0, second=0)

    assert Timestamp(**base).time_until == 1000  # not given: one second
    assert Timestamp(**base, time_until=0).time_until == 0  # given as zero: kept
    assert Timestamp(**base, time_until=None).time_until == 1000  # explicit None: absent
