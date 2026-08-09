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
