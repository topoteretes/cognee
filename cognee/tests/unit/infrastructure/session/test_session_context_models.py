import pytest
from pydantic import TypeAdapter, ValidationError

from cognee.infrastructure.session.session_context_models import (
    AGENT_VALID_SECTIONS,
    MAX_CONTEXT_CONTENT_CHARS,
    MIN_CANDIDATE_CONFIDENCE,
    VALID_PROFILES,
    AgentCandidateContextUpdate,
    AgentCandidateContextUpdateVariant,
    AgentContextExtraction,
    CandidateContextUpdate,
    ContextSection,
    ServedContextRating,
    SessionContextEntry,
    SessionFeedbackEntry,
    normalize_content,
    valid_sections_for,
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


# -- Profiles & agent sections ----------------------------------------------


def test_context_profile_and_agent_sections():
    assert VALID_PROFILES == {"qa", "agent"}
    assert AGENT_VALID_SECTIONS == {
        "tool_rules",
        "workflow_state",
        "success_patterns",
        "failure_lessons",
        "environment_facts",
    }
    assert valid_sections_for("qa") == {section.value for section in ContextSection}
    assert valid_sections_for("agent") == AGENT_VALID_SECTIONS
    assert valid_sections_for("nope") == set()


def test_session_context_entry_defaults_to_qa_profile():
    # Old rows carry no context_profile field — they must read as qa.
    entry = SessionContextEntry(id="1", section="rules", content="be concise", created_at="t")
    assert entry.context_profile == "qa"
    assert entry.source_trace_ids == []


def test_session_context_entry_agent_profile_valid():
    entry = SessionContextEntry(
        id="2",
        section="failure_lessons",
        context_profile="agent",
        content="sync before tests",
        created_at="t",
        source_trace_ids=[" trace-1 ", "", "trace-2"],
    )
    assert entry.context_profile == "agent"
    assert entry.section == "failure_lessons"
    assert entry.source_trace_ids == ["trace-1", "trace-2"]


@pytest.mark.parametrize(
    "profile,section",
    [("agent", "rules"), ("qa", "tool_rules"), ("agent", "goals"), ("qa", "failure_lessons")],
)
def test_session_context_entry_rejects_profile_section_mismatch(profile, section):
    with pytest.raises(ValidationError):
        SessionContextEntry(
            id="3", section=section, context_profile=profile, content="x", created_at="t"
        )


def test_session_context_entry_rejects_unknown_profile():
    with pytest.raises(ValidationError):
        SessionContextEntry(
            id="4", section="rules", context_profile="bogus", content="x", created_at="t"
        )


def test_agent_candidate_variants_round_trip():
    adapter = TypeAdapter(AgentCandidateContextUpdateVariant)
    for section in AGENT_VALID_SECTIONS:
        candidate = adapter.validate_python(
            {"section": section, "content": "lesson", "confidence": 0.9}
        )
        assert candidate.section == section
        assert candidate.context_profile == "agent"


def test_agent_candidate_rejects_qa_section():
    with pytest.raises(ValidationError):
        AgentCandidateContextUpdate(section="rules", content="x", confidence=0.9)


def test_qa_candidate_rejects_agent_section():
    with pytest.raises(ValidationError):
        CandidateContextUpdate(section="tool_rules", content="x", confidence=0.9)


def test_agent_context_extraction_normalizes_lessons():
    extraction = AgentContextExtraction(
        lessons=[
            {"section": " Tool_Rules ", "content": "Use uv run", "confidence": 0.9},
            {"section": "failure_lessons", "content": "Sync first", "confidence": 0.8},
        ]
    )
    assert [lesson.section for lesson in extraction.lessons] == ["tool_rules", "failure_lessons"]
    assert all(lesson.context_profile == "agent" for lesson in extraction.lessons)


def test_agent_context_extraction_defaults_to_empty():
    assert AgentContextExtraction().lessons == []
    assert AgentContextExtraction(lessons="not-a-list").lessons == []


def test_agent_context_extraction_lessons_use_base_model_not_discriminated_union():
    # AgentContextExtraction.lessons is intentionally typed as the plain base model, not the
    # AgentCandidateContextUpdateVariant discriminated union -- see the class docstring for why
    # (small local models via instructor are unreliable at picking a discriminator tag). Lock the
    # resulting item type so a future edit doesn't silently reintroduce the union.
    extraction = AgentContextExtraction(
        lessons=[{"section": "tool_rules", "content": "Use uv run", "confidence": 0.9}]
    )
    assert type(extraction.lessons[0]) is AgentCandidateContextUpdate


def test_agent_context_extraction_rejects_invalid_section():
    # The base model's own field_validator must still enforce AGENT_VALID_SECTIONS even without
    # a discriminator -- this is what makes dropping the union safe.
    with pytest.raises(ValidationError):
        AgentContextExtraction(
            lessons=[{"section": "not_a_real_section", "content": "x", "confidence": 0.9}]
        )


def test_agent_context_extraction_all_valid_sections_parse():
    extraction = AgentContextExtraction(
        lessons=[
            {"section": section, "content": "lesson", "confidence": 0.9}
            for section in AGENT_VALID_SECTIONS
        ]
    )
    assert {lesson.section for lesson in extraction.lessons} == AGENT_VALID_SECTIONS
    assert all(isinstance(lesson, AgentCandidateContextUpdate) for lesson in extraction.lessons)
