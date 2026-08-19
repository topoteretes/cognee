"""Unit coverage for the answer decision shared by the sequential and concurrent
session-turn paths (SDK-402): answer when there is a query to answer, when the
analysis found nothing worth acting on, or when there is no previous QA to be
feedback about; acknowledge otherwise.
"""

from cognee.infrastructure.session.feedback_models import SessionTurnAnalysis
from cognee.infrastructure.session.session_turn import decide_turn_answer


def test_answers_when_analysis_names_a_query_to_answer():
    analysis = SessionTurnAnalysis(
        response_to_user="Sure, on it.",
        query_to_answer="What is the capital of France?",
    )

    decision = decide_turn_answer(analysis, raw_query="raw", has_previous_qa=True)

    assert decision.should_answer is True
    assert decision.effective_query == "What is the capital of France?"


def test_answers_when_analysis_has_no_signal_at_all():
    decision = decide_turn_answer(SessionTurnAnalysis(), raw_query="raw", has_previous_qa=True)

    assert decision.should_answer is True
    assert decision.effective_query == "raw"


def test_answers_when_there_is_no_previous_qa_even_with_feedback_signal():
    analysis = SessionTurnAnalysis(response_to_user="Thanks!")

    decision = decide_turn_answer(analysis, raw_query="thanks!", has_previous_qa=False)

    assert decision.should_answer is True


def test_acknowledges_a_feedback_only_turn_with_a_previous_qa():
    analysis = SessionTurnAnalysis(response_to_user="Glad it helped!")

    decision = decide_turn_answer(analysis, raw_query="thanks!", has_previous_qa=True)

    assert decision.should_answer is False
    assert decision.response_to_user == "Glad it helped!"


def test_acknowledgement_defaults_to_got_it_when_analysis_gives_no_text():
    analysis = SessionTurnAnalysis(
        candidate_context_updates=[
            {"section": "rules", "content": "Always cite sources.", "confidence": 0.9}
        ]
    )

    decision = decide_turn_answer(analysis, raw_query="always cite sources", has_previous_qa=True)

    assert decision.should_answer is False
    assert decision.response_to_user == "Got it."
