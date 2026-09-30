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
        "July 1969",
        "1969-07-20 (Sunday)",
        "1940s",
        "20/07/1969",
        "1969-7-20",
        "1969-07-20 20:17",
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
