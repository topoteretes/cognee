"""Unit tests for the session operation-event mapper (pure, no cache/DB)."""

from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.modules.visualization.session_events import map_session_entries_to_events


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
