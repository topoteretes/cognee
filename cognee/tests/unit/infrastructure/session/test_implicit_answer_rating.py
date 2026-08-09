"""Tests for the implicit feedback_score bridge (turn analysis -> QA score).

This is what lets users who never call add_feedback still drive the graph
feedback-weights stage: clear sentiment about the previous answer becomes a
conservative implicit score on its QA entry.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_turn import (
    IMPLICIT_RATING_SCORES,
    apply_implicit_answer_rating,
)


def _session_manager():
    sm = MagicMock()
    sm.add_feedback = AsyncMock(return_value=True)
    return sm


@pytest.mark.asyncio
async def test_helpful_rating_writes_conservative_score():
    sm = _session_manager()

    applied = await apply_implicit_answer_rating(
        sm,
        user_id="u1",
        session_id="s1",
        previous_entry={"qa_id": "q1", "feedback_score": None},
        rating="helpful",
    )

    assert applied is True
    sm.add_feedback.assert_awaited_once_with(
        user_id="u1", session_id="s1", qa_id="q1", feedback_score=4
    )


@pytest.mark.asyncio
async def test_harmful_rating_writes_conservative_score():
    sm = _session_manager()

    await apply_implicit_answer_rating(
        sm,
        user_id="u1",
        session_id="s1",
        previous_entry={"qa_id": "q1"},
        rating="harmful",
    )

    assert sm.add_feedback.await_args.kwargs["feedback_score"] == 2


@pytest.mark.asyncio
async def test_explicit_score_is_never_overwritten():
    sm = _session_manager()

    applied = await apply_implicit_answer_rating(
        sm,
        user_id="u1",
        session_id="s1",
        previous_entry={"qa_id": "q1", "feedback_score": 5},
        rating="harmful",
    )

    assert applied is False
    sm.add_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_rating_or_missing_qa_is_a_noop():
    sm = _session_manager()

    assert not await apply_implicit_answer_rating(
        sm, user_id="u1", session_id="s1", previous_entry={"qa_id": "q1"}, rating=None
    )
    assert not await apply_implicit_answer_rating(
        sm, user_id="u1", session_id="s1", previous_entry={}, rating="helpful"
    )
    sm.add_feedback.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_feedback_failure_fails_open():
    sm = _session_manager()
    sm.add_feedback = AsyncMock(side_effect=RuntimeError("cache down"))

    applied = await apply_implicit_answer_rating(
        sm,
        user_id="u1",
        session_id="s1",
        previous_entry={"qa_id": "q1"},
        rating="helpful",
    )

    assert applied is False


def test_implicit_scores_are_conservative():
    """Implicit scores must stay off the 1/5 extremes reserved for explicit ratings."""
    assert set(IMPLICIT_RATING_SCORES.values()) <= {2, 3, 4}


def test_analysis_normalizes_overall_answer_rating():
    assert SessionTurnAnalysis(overall_answer_rating="  Helpful ").overall_answer_rating == (
        "helpful"
    )
    assert SessionTurnAnalysis(overall_answer_rating="meh").overall_answer_rating is None
    assert SessionTurnAnalysis(overall_answer_rating=None).overall_answer_rating is None
    assert SessionTurnAnalysis().overall_answer_rating is None


def test_analysis_normalizes_referenced_qa_ids():
    analysis = SessionTurnAnalysis(
        referenced_qa_ids=["q1", "", "  q2 ", 3, None, "q4", "q5", "q6", "q7"]
    )
    assert analysis.referenced_qa_ids == ["q1", "q2", "q4", "q5", "q6"]
    assert SessionTurnAnalysis(referenced_qa_ids="q1").referenced_qa_ids == []


def test_select_previous_answer_entry_skips_acknowledgements():
    from cognee.infrastructure.session.session_turn import select_previous_answer_entry

    real_answer = {
        "qa_id": "q1",
        "question": "what is X?",
        "answer": "X is ...",
        "context": "chunk",
        "used_session_context_ids": ["c1"],
    }
    acknowledgement = {
        "qa_id": "q2",
        "question": "that was wrong",
        "answer": "Got it.",
        "context": "",
    }

    assert select_previous_answer_entry([real_answer, acknowledgement]) is real_answer
    assert select_previous_answer_entry([acknowledgement]) is acknowledgement
    assert select_previous_answer_entry([]) == {}


def test_render_recent_turns_tags_qa_ids():
    from cognee.infrastructure.session.session_turn import render_recent_turns

    rendered = render_recent_turns(
        [
            {"qa_id": "q1", "question": "a?", "answer": "b"},
            {"qa_id": "q2", "question": "c?", "answer": "d"},
        ]
    )
    assert "[q1] User: a?" in rendered
    assert "[q2] Assistant: d" in rendered
