"""Unit tests for the session operation-event mapper and the collector's ordering
(no cache/DB: the session manager is stubbed)."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.visualization import session_events
from cognee.modules.visualization.session_events import map_session_entries_to_events

# importlib, not `import ... as`: the package __init__ rebinds the name to the
# function (see test_session_event_scoping.py).
get_session_manager_module = importlib.import_module(
    "cognee.infrastructure.session.get_session_manager"
)


def _entry(**overrides):
    base = {
        "time": "2026-06-11T10:00:00",
        "qa_id": "qa-1",
        "question": "Who is Alice?",
        "context": "ctx",
        "answer": "A curious girl.",
        "used_graph_element_ids": {"node_ids": ["n1", "n2"], "edge_ids": ["e1"]},
    }
    base.update(overrides)
    return SessionQAEntry(**base)


class TestMapSessionEntriesToEvents:
    def test_search_event_per_entry(self):
        events = map_session_entries_to_events("s1", [_entry()])
        assert len(events) == 1
        event = events[0]
        assert event["kind"] == "search"
        assert event["session_id"] == "s1"
        assert event["question"] == "Who is Alice?"
        assert event["answer"] == "A curious girl."
        assert event["node_ids"] == ["n1", "n2"]
        assert event["edge_ids"] == ["e1"]

    def test_rated_entry_also_emits_improve_event(self):
        entry = _entry(
            feedback_score=5,
            feedback_text="great answer",
            memify_metadata={"feedback_weights_applied": True},
        )
        events = map_session_entries_to_events("s1", [entry])
        assert [e["kind"] for e in events] == ["search", "improve"]
        improve = events[1]
        assert improve["rating"] == 5
        assert improve["feedback_text"] == "great answer"
        assert improve["applied"] is True
        # Reinforcement targets the same elements the search retrieved.
        assert improve["node_ids"] == ["n1", "n2"]

    def test_unapplied_feedback_marked_pending(self):
        events = map_session_entries_to_events("s1", [_entry(feedback_score=2)])
        assert events[1]["applied"] is False

    def test_entries_without_provenance_still_map_when_question_present(self):
        events = map_session_entries_to_events("s1", [_entry(used_graph_element_ids=None)])
        assert len(events) == 1
        assert events[0]["node_ids"] == []

    def test_placeholder_entries_skipped(self):
        events = map_session_entries_to_events(
            "s1", [_entry(question="", used_graph_element_ids=None)]
        )
        assert events == []

    def test_determinism(self):
        entries = [_entry(), _entry(qa_id="qa-2", feedback_score=4)]
        assert map_session_entries_to_events("s1", entries) == map_session_entries_to_events(
            "s1", entries
        )


@pytest.mark.asyncio
async def test_collected_events_are_ordered_by_utc_time_not_by_string():
    """`11:00:10+02:00` is 09:00:10 UTC, earlier than a naive `10:00:00`, though
    it sorts after it as text. The order matters beyond display: live-events
    takes the last event as its cursor, so a misplaced event would move the
    cursor back and redeliver the newer one on the next poll."""
    sessions = {
        "s1": [_entry(qa_id="naive-newer", time="2026-08-03T10:00:00.000000", feedback_score=4)],
        "s2": [
            _entry(qa_id="offset-older", time="2026-08-03T11:00:10+02:00"),
            _entry(qa_id="untimed", time=""),
        ],
    }

    async def _get_session(*, user_id, session_id):
        return sessions[session_id]

    with patch.object(
        get_session_manager_module,
        "get_session_manager",
        return_value=SimpleNamespace(is_available=True, get_session=_get_session),
    ):
        events = await session_events.collect_session_events(
            user=SimpleNamespace(id="user"), session_ids=["s1", "s2"]
        )

    assert [(e["qa_id"], e["kind"]) for e in events] == [
        ("untimed", "search"),
        ("offset-older", "search"),
        ("naive-newer", "search"),
        ("naive-newer", "improve"),
    ]
