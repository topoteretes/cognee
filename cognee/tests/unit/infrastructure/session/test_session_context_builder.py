"""Unit tests for the deterministic session-context builder, ranker, and candidate applier.

These tests use FAKE session managers (plain in-memory objects, no real cache / Redis) so they
run anywhere. They verify ranker ordering, budget capping, candidate validation/dedup, and the
fail-open contract.
"""

from datetime import datetime
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_context_builder import (
    DeterministicRanker,
    apply_candidate_updates,
    build_active_context_block,
)
from cognee.infrastructure.session.session_context_models import (
    SessionContextEntry,
    normalize_content,
)


def _entry(section, content, **kwargs):
    return SessionContextEntry(
        id=kwargs.pop("id", str(uuid4())),
        section=section,
        content=content,
        normalized_content=normalize_content(content),
        created_at=kwargs.pop("created_at", datetime.utcnow().isoformat()),
        **kwargs,
    )


class FakeSessionManager:
    """In-memory stand-in for SessionManager's context CRUD surface."""

    def __init__(self, entries=None):
        # Store as dicts to mirror the real cache payload shape.
        self.store = [
            e.model_dump() if isinstance(e, SessionContextEntry) else e for e in (entries or [])
        ]

    async def get_session_context_entries(self, user_id, session_id):
        return list(self.store)

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        self.store.append(entry_dump)

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False


class RaisingSessionManager:
    """Every call raises, to exercise the fail-open paths."""

    async def get_session_context_entries(self, user_id, session_id):
        raise RuntimeError("boom")

    async def create_session_context_entry(self, user_id, session_id, entry_dump):
        raise RuntimeError("boom")

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        raise RuntimeError("boom")


class UpdateRaisingSessionManager(FakeSessionManager):
    """Loads entries normally but fails when stamping served entries."""

    async def update_session_context_entry(self, user_id, session_id, entry_id, merge):
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- ranker


def test_ranker_section_priority_rules_over_goals():
    ranker = DeterministicRanker()
    rule = _entry("rules", "always use tabs")
    goal = _entry("goals", "ship the feature")
    assert ranker.score(rule, "anything") > ranker.score(goal, "anything")


def test_ranker_confidence_breaks_within_section():
    ranker = DeterministicRanker()
    high = _entry("goals", "high confidence goal", confidence=0.95)
    low = _entry("goals", "low confidence goal", confidence=0.1)
    assert ranker.score(high, "unrelated") > ranker.score(low, "unrelated")


def test_ranker_query_overlap_boosts():
    ranker = DeterministicRanker()
    entry = _entry("preferences", "use postgres database")
    overlap_score = ranker.score(entry, "how do I configure postgres database")
    no_overlap_score = ranker.score(entry, "completely different topic")
    assert overlap_score > no_overlap_score


def test_ranker_net_helpfulness():
    ranker = DeterministicRanker()
    helpful = _entry("goals", "g", helpful_count=5, harmful_count=0)
    harmful = _entry("goals", "g", helpful_count=0, harmful_count=5)
    assert ranker.score(helpful, "x") > ranker.score(harmful, "x")


def test_ranker_clamps_net_helpfulness_below_section_priority():
    ranker = DeterministicRanker()
    rule = _entry("rules", "rule", confidence=0.0)
    preference = _entry("preferences", "preference", helpful_count=100, confidence=0.0)
    assert ranker.score(rule, "x") > ranker.score(preference, "x")


# --------------------------------------------------------------------------- builder


@pytest.mark.asyncio
async def test_build_empty_returns_empty():
    sm = FakeSessionManager([])
    block, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="q"
    )
    assert block == ""
    assert served == []


@pytest.mark.asyncio
async def test_build_renders_sections_and_records_ids():
    entries = [
        _entry("goals", "Ship the MVP", id="g1"),
        _entry("rules", "Always run ruff", id="r1"),
        _entry("preferences", "Prefer uv", id="p1"),
        _entry("lessons_learned", "Redis unavailable locally", id="l1"),
    ]
    sm = FakeSessionManager(entries)
    block, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="anything"
    )
    assert "## Active session guidance" in block
    assert "### Goals" in block
    assert "### Rules" in block
    assert "### Preferences" in block
    assert "### Lessons learned" in block
    assert set(served) == {"g1", "r1", "p1", "l1"}
    for row in sm.store:
        assert row["last_served_at"]


@pytest.mark.asyncio
async def test_build_renders_same_section_oldest_to_newest_with_time_labels():
    entries = [
        _entry(
            "preferences",
            "Prefer 4 concise bullet points.",
            id="p-new",
            created_at="2026-06-10T10:19:08",
        ),
        _entry(
            "preferences",
            "Prefer 2 informative bullet points.",
            id="p-old",
            created_at="2026-06-10T10:14:22",
        ),
    ]
    sm = FakeSessionManager(entries)

    block, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="bullet points"
    )

    assert "When guidance conflicts, prefer the later item." in block
    assert block.index("[10:14:22] Prefer 2 informative bullet points.") < block.index(
        "[10:19:08] Prefer 4 concise bullet points."
    )
    assert set(served) == {"p-old", "p-new"}


@pytest.mark.asyncio
async def test_build_returns_block_when_served_stamp_fails():
    entries = [_entry("goals", "Ship the MVP", id="g1")]
    sm = UpdateRaisingSessionManager(entries)
    block, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="anything"
    )
    assert "Ship the MVP" in block
    assert served == ["g1"]
    assert sm.store[0]["last_served_at"] is None


@pytest.mark.asyncio
async def test_build_total_budget_caps_output():
    # Many long entries; total budget should drop some.
    entries = [_entry("goals", "x" * 100, id=f"g{i}") for i in range(20)]
    sm = FakeSessionManager(entries)
    block, served = await build_active_context_block(
        session_manager=sm,
        user_id="u",
        session_id="s",
        query="q",
        per_section_char_budget=10_000,
        total_char_budget=300,
    )
    # 300 / 100 chars = at most 3 entries fit.
    assert len(served) <= 3
    assert len(block) < 600


@pytest.mark.asyncio
async def test_build_per_section_budget_caps_output():
    entries = [_entry("goals", "y" * 100, id=f"g{i}") for i in range(10)]
    sm = FakeSessionManager(entries)
    _, served = await build_active_context_block(
        session_manager=sm,
        user_id="u",
        session_id="s",
        query="q",
        per_section_char_budget=250,
        total_char_budget=10_000,
    )
    assert len(served) <= 2


@pytest.mark.asyncio
async def test_build_skips_feedback_kind_entries():
    ctx = _entry("goals", "real goal", id="g1")
    feedback_row = {"id": "f1", "kind": "feedback", "section": "goals", "content": "fb"}
    sm = FakeSessionManager([ctx])
    sm.store.append(feedback_row)
    _, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="q"
    )
    assert served == ["g1"]


@pytest.mark.asyncio
async def test_build_fail_open_on_raising_manager():
    sm = RaisingSessionManager()
    block, served = await build_active_context_block(
        session_manager=sm, user_id="u", session_id="s", query="q"
    )
    assert block == ""
    assert served == []


# --------------------------------------------------------------------------- applier


@pytest.mark.asyncio
async def test_apply_creates_new_for_novel_content():
    sm = FakeSessionManager([])
    candidates = [{"section": "rules", "content": "Always use double quotes", "confidence": 0.9}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb1",
        candidates=candidates,
    )
    assert len(touched) == 1
    assert len(sm.store) == 1
    created = sm.store[0]
    assert created["section"] == "rules"
    assert created["normalized_content"] == "always use double quotes"
    assert created["source_feedback_ids"] == ["fb1"]


@pytest.mark.asyncio
async def test_apply_rejects_low_confidence():
    sm = FakeSessionManager([])
    candidates = [{"section": "rules", "content": "maybe do this", "confidence": 0.5}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb1",
        candidates=candidates,
    )
    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_apply_rejects_empty_content():
    sm = FakeSessionManager([])
    # Pydantic validator on CandidateContextUpdate strips; empty content -> skipped fail-open.
    candidates = [{"section": "goals", "content": "   ", "confidence": 0.9}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb1",
        candidates=candidates,
    )
    assert touched == []
    assert sm.store == []


@pytest.mark.asyncio
async def test_apply_links_exact_duplicate_no_new_entry():
    existing = _entry("rules", "Always use double quotes", id="r1")
    sm = FakeSessionManager([existing])
    candidates = [{"section": "rules", "content": "always use DOUBLE quotes", "confidence": 0.9}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb2",
        candidates=candidates,
    )
    assert touched == ["r1"]
    # No new entry created.
    assert len(sm.store) == 1
    assert sm.store[0]["source_feedback_ids"] == ["fb2"]


@pytest.mark.asyncio
async def test_apply_fail_open_on_raising_manager():
    sm = RaisingSessionManager()
    candidates = [{"section": "rules", "content": "do something", "confidence": 0.9}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb1",
        candidates=candidates,
    )
    assert touched == []


@pytest.mark.asyncio
async def test_apply_handles_invalid_candidate_dict():
    sm = FakeSessionManager([])
    # Invalid section -> CandidateContextUpdate validation raises -> skipped fail-open.
    candidates = [{"section": "not_a_section", "content": "x", "confidence": 0.9}]
    touched = await apply_candidate_updates(
        session_manager=sm,
        user_id="u",
        session_id="s",
        feedback_entry_id="fb1",
        candidates=candidates,
    )
    assert touched == []
    assert sm.store == []
