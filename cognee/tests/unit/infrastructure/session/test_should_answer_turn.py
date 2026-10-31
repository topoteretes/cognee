"""Unit coverage for the answer predicate shared by the sequential and concurrent
session-turn paths (SDK-402): answer when there is a query to answer, when the
analysis found nothing worth acting on, or when there is no previous QA to be
feedback about; acknowledge otherwise.
"""

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_turn import (
    acknowledgement_for_turn,
    should_answer_turn,
)


def test_answers_when_analysis_names_a_query_to_answer():
    analysis = SessionTurnAnalysis(
        response_to_user="Sure, on it.",
        query_to_answer="What is the capital of France?",
    )

    assert should_answer_turn(analysis, has_previous_qa=True) is True


def test_answers_when_analysis_has_no_signal_at_all():
    assert should_answer_turn(SessionTurnAnalysis(), has_previous_qa=True) is True


def test_answers_when_there_is_no_previous_qa_even_with_feedback_signal():
    analysis = SessionTurnAnalysis(response_to_user="Thanks!")

    assert should_answer_turn(analysis, has_previous_qa=False) is True


def test_acknowledges_a_feedback_only_turn_with_a_previous_qa():
    analysis = SessionTurnAnalysis(response_to_user="Glad it helped!")

    assert should_answer_turn(analysis, has_previous_qa=True) is False
    assert acknowledgement_for_turn(analysis.response_to_user) == "Glad it helped!"


def test_acknowledgement_defaults_to_got_it_when_analysis_gives_no_text():
    analysis = SessionTurnAnalysis(
        candidate_context_updates=[
            {"section": "rules", "content": "Always cite sources.", "confidence": 0.9}
        ]
    )

    assert should_answer_turn(analysis, has_previous_qa=True) is False
    assert acknowledgement_for_turn(analysis.response_to_user) == "Got it."
