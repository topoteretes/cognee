from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.session.session_context_models import MAX_CONTEXT_CONTENT_CHARS
from cognee.infrastructure.session.session_search_models import (
    MaintenanceApplyResult,
    SessionMaintenanceResult,
    SessionMaintenanceWorkItem,
    SessionSearchCompletion,
    SessionTurnEvidence,
    SessionTurnSnapshot,
    get_session_search_completion_model,
)


class Answer(BaseModel):
    text: str


def test_string_completion_model_preserves_string_response():
    model = get_session_search_completion_model(str)

    result = model(response="answer")

    assert result.response == "answer"
    assert isinstance(result, SessionSearchCompletion)


def test_custom_completion_model_converts_and_validates_response_kind():
    model = get_session_search_completion_model(Answer)

    result = model(response={"text": "answer"})

    assert result.response == Answer(text="answer")
    with pytest.raises(ValidationError):
        model(response="wrong type")
    with pytest.raises(ValidationError):
        model(response={"text": "wrong acknowledgement"}, is_acknowledgement=True)


def test_acknowledgement_requires_a_string_for_custom_models():
    model = get_session_search_completion_model(Answer)

    result = model(response="Understood.", is_acknowledgement=True)

    assert result.response == "Understood."


def test_completion_model_factory_is_cached_and_rejects_other_types():
    assert get_session_search_completion_model(Answer) is get_session_search_completion_model(
        Answer
    )
    with pytest.raises(TypeError):
        get_session_search_completion_model(dict)


def test_completion_evidence_is_cleaned_capped_and_bounded():
    model = get_session_search_completion_model(str)

    result = model(
        response="answer",
        feedback_evidence=[" first ", 3, "x" * 500, "third", "fourth"],
        future_context_evidence="malformed",
    )

    assert result.feedback_evidence == [
        "first",
        "x" * MAX_CONTEXT_CONTENT_CHARS,
        "third",
    ]
    assert result.future_context_evidence == []


def test_snapshot_and_work_item_are_immutable():
    snapshot = SessionTurnSnapshot(raw_message="question")
    work_item = SessionMaintenanceWorkItem(
        evidence_id="e1",
        user_id="u1",
        session_id="s1",
        llm_config=LLMConfig(),
    )

    with pytest.raises(ValidationError):
        snapshot.raw_message = "changed"
    with pytest.raises(ValidationError):
        work_item.evidence_id = "changed"


def test_turn_evidence_has_storage_defaults_and_bounds_untrusted_fields():
    evidence = SessionTurnEvidence(
        id="e1",
        created_at=datetime.now(timezone.utc).isoformat(),
        current_raw_message="question",
        current_response="answer",
        feedback_evidence="malformed",
        future_context_evidence=[" keep ", object()],
        error="x" * 500,
    )

    assert evidence.kind == "turn_evidence"
    assert evidence.status == "pending"
    assert evidence.feedback_evidence == []
    assert evidence.future_context_evidence == ["keep"]
    assert len(evidence.error) == MAX_CONTEXT_CONTENT_CHARS


def test_maintenance_contracts_cap_results_and_errors():
    result = SessionMaintenanceResult(
        served_context_ratings=[
            {"entry_id": f"ctx-{index}", "rating": "helpful"} for index in range(5)
        ],
        candidate_context_updates=[
            {"section": "rules", "content": f"rule {index}", "confidence": 0.9}
            for index in range(5)
        ],
    )
    apply_result = MaintenanceApplyResult(errors=["x" * 500, "two", "three", "four"])

    assert len(result.served_context_ratings) == 3
    assert len(result.candidate_context_updates) == 3
    assert len(apply_result.errors) == 3
    assert len(apply_result.errors[0]) == MAX_CONTEXT_CONTENT_CHARS
