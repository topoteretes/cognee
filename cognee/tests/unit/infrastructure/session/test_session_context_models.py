import pytest
from pydantic import ValidationError

from cognee.infrastructure.session.session_context_models import (
    MAX_CONTEXT_CONTENT_CHARS,
    MIN_CANDIDATE_CONFIDENCE,
    CandidateContextUpdate,
    ContextSection,
    ServedContextRating,
    SessionContextEntry,
    SessionFeedbackEntry,
    normalize_content,
)


def test_constants():
    assert MIN_CANDIDATE_CONFIDENCE == 0.75
    assert MAX_CONTEXT_CONTENT_CHARS == 280


def test_context_section_enum_values():
    assert ContextSection.GOALS.value == "goals"
    assert ContextSection.RULES.value == "rules"
    assert ContextSection.PREFERENCES.value == "preferences"
    assert ContextSection.LESSONS_LEARNED.value == "lessons_learned"


def test_normalize_content():
    assert normalize_content("  Use  TABS ") == "use tabs"
    assert normalize_content("Already normal") == "already normal"
    assert normalize_content("\tLeading\n  white\t space\n") == "leading white space"
    assert normalize_content("") == ""


def test_served_context_rating_valid():
    rating = ServedContextRating(entry_id="abc", rating="Helpful")
    assert rating.rating == "helpful"
    rating2 = ServedContextRating(entry_id=" id ", rating=" HARMFUL ")
    assert rating2.rating == "harmful"
    assert rating2.entry_id == "id"


def test_served_context_rating_invalid_rating():
    with pytest.raises(ValidationError):
        ServedContextRating(entry_id="abc", rating="meh")


def test_served_context_rating_empty_entry_id():
    with pytest.raises(ValidationError):
        ServedContextRating(entry_id="   ", rating="helpful")


def test_candidate_context_update_valid_and_section_normalization():
    candidate = CandidateContextUpdate(section=" Rules ", content="  Be concise  ", confidence=0.9)
    assert candidate.section == "rules"
    assert candidate.content == "Be concise"
    assert candidate.confidence == 0.9


def test_candidate_context_update_invalid_section():
    with pytest.raises(ValidationError):
        CandidateContextUpdate(section="bogus", content="x")


def test_candidate_content_truncation():
    long = "a" * (MAX_CONTEXT_CONTENT_CHARS + 50)
    candidate = CandidateContextUpdate(section="goals", content=long)
    assert len(candidate.content) == MAX_CONTEXT_CONTENT_CHARS


def test_candidate_confidence_clamp():
    assert CandidateContextUpdate(section="goals", content="x", confidence=2.5).confidence == 1.0
    assert CandidateContextUpdate(section="goals", content="x", confidence=-1.0).confidence == 0.0
    assert CandidateContextUpdate(section="goals", content="x").confidence == 0.0


def test_session_context_entry_valid_and_kind_default():
    entry = SessionContextEntry(
        id="1",
        section="preferences",
        content="Prefer markdown",
        normalized_content="prefer markdown",
        created_at="2026-06-05T00:00:00",
    )
    assert entry.kind == "context"
    assert entry.section == "preferences"
    assert entry.confidence == 0.0
    assert entry.source_feedback_ids == []
    assert entry.helpful_count == 0
    assert entry.last_served_at is None


def test_session_context_entry_invalid_section():
    with pytest.raises(ValidationError):
        SessionContextEntry(
            id="1",
            section="nope",
            content="x",
            normalized_content="x",
            created_at="t",
        )


def test_session_context_entry_content_truncation():
    long = "b" * (MAX_CONTEXT_CONTENT_CHARS + 100)
    entry = SessionContextEntry(
        id="1",
        section="rules",
        content=long,
        normalized_content="b",
        created_at="t",
    )
    assert len(entry.content) == MAX_CONTEXT_CONTENT_CHARS


def test_session_context_entry_empty_content_raises():
    with pytest.raises(ValidationError):
        SessionContextEntry(
            id="1",
            section="rules",
            content="   ",
            normalized_content="",
            created_at="t",
        )


def test_session_context_entry_confidence_clamp():
    entry = SessionContextEntry(
        id="1",
        section="goals",
        content="x",
        normalized_content="x",
        created_at="t",
        confidence=5.0,
    )
    assert entry.confidence == 1.0


def test_session_context_entry_negative_count_raises():
    with pytest.raises(ValidationError):
        SessionContextEntry(
            id="1",
            section="goals",
            content="x",
            normalized_content="x",
            created_at="t",
            helpful_count=-1,
        )


def test_session_feedback_entry_kind_default():
    entry = SessionFeedbackEntry(id="f1", created_at="t", raw_text="that was wrong")
    dump = entry.model_dump()
    assert entry.kind == "feedback"
    assert entry.referenced_qa_ids == []
    assert entry.influencing_context_ids == []
    assert entry.candidate_context_entries == []
    assert "feedback_text" not in dump
    assert "feedback_score" not in dump
