"""Timestamp precision: an extracted "1996" means the whole year, not its first second."""

from cognee.modules.engine.utils.generate_event_datapoint import generate_event_datapoint
from cognee.tasks.temporal_graph.models import Event as ExtractedEvent
from cognee.tasks.temporal_graph.models import Timestamp
from cognee.tasks.temporal_graph.time_precision import expand_to_period_end, infer_precision


def test_infer_precision_from_defaults():
    assert infer_precision(Timestamp(year=1996)) == "year"
    assert infer_precision(Timestamp(year=1996, month=3)) == "month"
    assert infer_precision(Timestamp(year=1996, month=3, day=5)) == "day"
    assert infer_precision(Timestamp(year=1996, month=3, day=5, hour=7)) == "hour"
    assert infer_precision(Timestamp(year=1996, month=1, day=1, minute=1)) == "minute"
    assert infer_precision(Timestamp(year=1996, second=1)) == "second"


def test_explicit_precision_wins():
    assert infer_precision(Timestamp(year=1996, precision="day")) == "day"


def test_expand_year_month_day():
    year = expand_to_period_end(Timestamp(year=1996))
    assert (year.month, year.day, year.hour, year.minute, year.second) == (12, 31, 23, 59, 59)
    month = expand_to_period_end(Timestamp(year=2024, month=2))
    assert (month.day, month.hour) == (29, 23)  # leap day
    day = expand_to_period_end(Timestamp(year=2022, month=3, day=5))
    assert (day.day, day.hour, day.minute, day.second) == (5, 23, 59, 59)
    assert expand_to_period_end(None) is None


def test_event_interval_end_is_widened_but_instants_are_kept():
    interval_event = generate_event_datapoint(
        ExtractedEvent(
            name="Pahang spell",
            time_from=Timestamp(year=1994),
            time_to=Timestamp(year=1996),
        )
    )
    assert interval_event.during.time_from.timestamp_str == "1994-01-01 00:00:00"
    assert interval_event.during.time_to.timestamp_str == "1996-12-31 23:59:59"

    instant_event = generate_event_datapoint(
        ExtractedEvent(name="Retired", time_from=Timestamp(year=1998))
    )
    assert instant_event.at.timestamp_str == "1998-01-01 00:00:00"
    assert instant_event.during is None

    end_only_event = generate_event_datapoint(
        ExtractedEvent(name="Ended", time_to=Timestamp(year=1998))
    )
    assert end_only_event.at.timestamp_str == "1998-01-01 00:00:00"
