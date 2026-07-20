"""Unit tests for session operation-event collection and mapping."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.session.session_scope import get_storage_session_id
from cognee.modules.session_lifecycle.models import SessionRecord
from cognee.modules.visualization.session_events import (
    _list_recent_session_ids,
    map_session_entries_to_events,
)


class _RelationalEngine:
    def __init__(self, engine):
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    def get_async_session(self):
        return self._sessions()


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
async def test_recent_session_ids_are_public_and_filtered_to_active_dataset(monkeypatch):
    import cognee.infrastructure.databases.relational as relational

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SessionRecord.__table__.create)
    wrapper = _RelationalEngine(engine)
    monkeypatch.setattr(relational, "get_relational_engine", lambda: wrapper)

    owner_id = uuid4()
    other_owner_id = uuid4()
    dataset_a = uuid4()
    dataset_b = uuid4()
    now = datetime.now(timezone.utc)

    async with wrapper.get_async_session() as session:
        session.add_all(
            [
                SessionRecord(
                    session_id=get_storage_session_id("dataset-a-session", dataset_a),
                    public_session_id="dataset-a-session",
                    user_id=owner_id,
                    dataset_id=dataset_a,
                    status="running",
                    started_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id=get_storage_session_id("dataset-b-session", dataset_b),
                    public_session_id="dataset-b-session",
                    user_id=owner_id,
                    dataset_id=dataset_b,
                    status="running",
                    started_at=now,
                    last_activity_at=now,
                ),
                # A legacy row may carry a stale sticky dataset id, but must
                # remain quarantined from dataset-scoped visualization.
                SessionRecord(
                    session_id="legacy-session",
                    public_session_id=None,
                    user_id=owner_id,
                    dataset_id=dataset_a,
                    status="running",
                    started_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id=get_storage_session_id("other-owner-session", dataset_a),
                    public_session_id="other-owner-session",
                    user_id=other_owner_id,
                    dataset_id=dataset_a,
                    status="running",
                    started_at=now,
                    last_activity_at=now,
                ),
            ]
        )
        await session.commit()

    try:
        dataset_token = current_dataset_id.set(str(dataset_a))
        try:
            assert await _list_recent_session_ids(owner_id, 10) == ["dataset-a-session"]
        finally:
            current_dataset_id.reset(dataset_token)

        legacy_token = current_dataset_id.set(None)
        try:
            assert await _list_recent_session_ids(owner_id, 10) == ["legacy-session"]
        finally:
            current_dataset_id.reset(legacy_token)
    finally:
        await engine.dispose()
