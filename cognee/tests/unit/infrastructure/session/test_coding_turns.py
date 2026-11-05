"""Unit tests for pairing a coding session's traces to its turns (SDK-1335)."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cognee.infrastructure.session.coding_turns import (
    build_turns,
    group_by_turn,
    owning_turn_index,
    parse_event_time,
    settled_traces,
)


def _row(qa_id, time, prompt_time="", **context):
    return {
        "qa_id": qa_id,
        "question": f"request {qa_id}",
        "answer": f"answer {qa_id}",
        "time": time,
        "prompt_time": prompt_time,
        "context": json.dumps(context) if context else "",
    }


def _trace(trace_id, time=""):
    return SimpleNamespace(trace_id=trace_id, time=time)


T = "2026-11-05T10:{:02d}:00+00:00".format


def test_parse_event_time_reads_aware_naive_and_zulu_and_rejects_the_rest():
    assert parse_event_time("2026-11-05T10:00:00+02:00") == datetime(
        2026, 11, 5, 8, 0, tzinfo=timezone.utc
    )
    assert parse_event_time("2026-11-05T10:00:00") == datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
    assert parse_event_time("2026-11-05T10:00:00Z") == datetime(
        2026, 11, 5, 10, tzinfo=timezone.utc
    )
    assert parse_event_time("") is None and parse_event_time(None) is None
    assert parse_event_time("yesterday") is None


def test_turns_are_in_time_order_and_start_at_their_prompt():
    turns = build_turns(
        [
            _row("q2", T(9), prompt_time=T(6)),
            _row("q1", T(5), prompt_time=T(0), cwd="/repo"),
        ]
    )
    assert [turn.qa_id for turn in turns] == ["q1", "q2"]
    assert turns[0].start == parse_event_time(T(0)) and turns[0].end == parse_event_time(T(5))
    assert turns[0].qa_context == {"cwd": "/repo"}
    assert turns[1].request == "request q2" and turns[1].final_answer == "answer q2"


def test_without_a_prompt_time_a_turn_starts_at_the_previous_answer():
    turns = build_turns([_row("q1", T(5)), _row("q2", T(9))])
    assert turns[0].start is None
    assert turns[1].start == parse_event_time(T(5))


def test_a_trace_belongs_to_the_latest_turn_started_before_it():
    turns = build_turns([_row("q1", T(5), prompt_time=T(0)), _row("q2", T(9), prompt_time=T(6))])
    assert owning_turn_index(turns, _trace("t", T(1))) == 0
    assert owning_turn_index(turns, _trace("t", T(7))) == 1
    # Between the first answer and the next prompt: the first turn's late work.
    assert owning_turn_index(turns, _trace("t", "2026-11-05T10:05:30+00:00")) == 0
    # Before every start: the first turn. No time: the newest turn.
    assert owning_turn_index(turns, _trace("t", "2026-11-04T00:00:00+00:00")) == 0
    assert owning_turn_index(turns, _trace("t")) == 1
    with pytest.raises(ValueError):
        owning_turn_index([], _trace("t", T(1)))


def test_settled_traces_stop_at_the_first_trace_newer_than_the_last_answer():
    turns = build_turns([_row("q1", T(5)), _row("q2", T(9))])
    traces = [_trace("a", T(1)), _trace("b", T(7)), _trace("new", T(10)), _trace("replayed", T(8))]
    assert [trace.trace_id for trace in settled_traces(traces, turns)] == ["a", "b"]
    assert settled_traces(traces, []) == []
    assert [t.trace_id for t in settled_traces([_trace("untimed")], turns)] == ["untimed"]


def test_group_by_turn_keeps_arrival_order_within_a_turn_and_orders_turns():
    turns = build_turns([_row("q1", T(5), prompt_time=T(0)), _row("q2", T(9), prompt_time=T(6))])
    # Arrival order: the second turn's first trace lands before the first turn's last one.
    traces = [_trace("a", T(1)), _trace("c", T(7)), _trace("b", "2026-11-05T10:05:30+00:00")]
    grouped = group_by_turn(traces, turns)
    assert [(turn.qa_id, [t.trace_id for t in turn.traces]) for turn in grouped] == [
        ("q1", ["a", "b"]),
        ("q2", ["c"]),
    ]
    assert group_by_turn([], turns) == []
