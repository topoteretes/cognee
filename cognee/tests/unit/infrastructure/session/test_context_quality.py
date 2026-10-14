from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.session.context_quality import supported_project_candidate, unused_entry
from cognee.infrastructure.session.session_context_builder import apply_candidate_updates
from cognee.infrastructure.session.session_context_models import (
    CandidateEnvironmentFactUpdate,
    SessionContextEntry,
)


def candidate(content="The outbox cursor commits with delivery to prevent event loss.", **kw):
    return CandidateEnvironmentFactUpdate(content=content, confidence=0.9, **kw)


def test_requires_stated_support_and_rejects_generic_tool_advice():
    assert not supported_project_candidate(candidate(), "trace")
    assert supported_project_candidate(
        candidate(reason="Separate commits lose events after a crash."), "trace"
    )
    assert not supported_project_candidate(
        candidate(
            "Parse Bash output before continuing.",
            evidence="A Bash command returned JSON in this trace.",
        ),
        "trace",
    )


@pytest.mark.asyncio
async def test_unsupported_candidate_never_reaches_storage():
    manager = AsyncMock()
    ids = await apply_candidate_updates(
        session_manager=manager,
        user_id="u",
        session_id="s",
        source_id="trace",
        candidates=[candidate()],
    )
    assert ids == []
    manager.create_session_context_entry.assert_not_called()


def test_unused_sessions_cutoff_preserves_recently_recalled_and_qa_entries():
    entry = SessionContextEntry(
        id="e",
        section="environment_facts",
        context_profile="agent",
        content="fact",
        confidence=0.9,
        created_at="2026-01-01T00:00:00+00:00",
    )
    cutoff = datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert unused_entry(entry, cutoff)
    assert not unused_entry(
        entry.model_copy(update={"last_served_at": "2026-02-02T00:00:00+00:00"}), cutoff
    )
    assert not unused_entry(entry.model_copy(update={"context_profile": "qa"}), cutoff)


@pytest.mark.asyncio
async def test_decay_counts_only_same_user_and_dataset_sessions(monkeypatch):
    from uuid import uuid4

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.session.context_quality import filter_unused_entries
    from cognee.modules.session_lifecycle.models import SessionRecord

    uid, other, dataset, unrelated = uuid4(), uuid4(), uuid4(), uuid4()
    old = "quality-" + uuid4().hex
    monkeypatch.setenv("COGNEE_AGENT_CONTEXT_UNUSED_SESSIONS", "2")
    engine = get_relational_engine()
    jan = datetime(2026, 1, 1, tzinfo=timezone.utc)
    feb = datetime(2026, 2, 1, tzinfo=timezone.utc)
    async with engine.get_async_session() as db:
        db.add(SessionRecord(user_id=uid, session_id=old, dataset_id=dataset, started_at=jan))
        for owner, ds in [(other, dataset), (uid, unrelated)]:
            for _ in range(3):
                db.add(
                    SessionRecord(
                        user_id=owner, session_id=uuid4().hex, dataset_id=ds, started_at=feb
                    )
                )
        await db.commit()
    entry = SessionContextEntry(
        id="e",
        section="environment_facts",
        context_profile="agent",
        content="fact",
        confidence=0.9,
        created_at=jan.isoformat(),
    )
    manager = AsyncMock()
    assert await filter_unused_entries(manager, str(uid), old, [entry]) == [entry]
    async with engine.get_async_session() as db:
        for _ in range(2):
            db.add(
                SessionRecord(
                    user_id=uid, session_id=uuid4().hex, dataset_id=dataset, started_at=feb
                )
            )
        await db.commit()
    assert await filter_unused_entries(manager, str(uid), old, [entry]) == []
    manager.update_session_context_entry.assert_not_called()
    assert await filter_unused_entries(manager, str(uid), old, [entry], persist=True) == []
    assert manager.update_session_context_entry.await_args.kwargs["merge"]["expired_at"]
