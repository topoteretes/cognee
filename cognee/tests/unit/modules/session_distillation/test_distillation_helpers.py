"""Unit tests for the deterministic parts of session distillation.

The gate, the session digest, evidence-pack selection, and the document renderer are
pure functions, so they are tested directly with fixtures — no LLM, cache, or vector
engine involved.
"""

from uuid import uuid4

from cognee.modules.session_distillation.distill import (
    build_session_digest,
    build_source_messages_by_entry,
    gate_context_entries,
    render_distilled_document,
    select_evidence_excerpts,
)
from cognee.modules.session_distillation.models import (
    MAX_EVIDENCE_EXCERPTS,
    CuratedLesson,
    DistilledLesson,
)


def _context_row(section="lessons_learned", content="A lesson.", **overrides):
    row = {
        "id": str(uuid4()),
        "section": section,
        "content": content,
        "normalized_content": content.lower(),
        "confidence": 0.9,
        "created_at": "2026-06-11T10:00:00",
        "source_feedback_ids": [],
        "helpful_count": 0,
        "harmful_count": 0,
        "priority": 0,
        "kind": "context",
    }
    row.update(overrides)
    return row


def _qa_row(question="What?", answer="That.", **overrides):
    row = {
        "time": "2026-06-11T10:00:00",
        "qa_id": str(uuid4()),
        "question": question,
        "context": "",
        "answer": answer,
        "used_session_context_ids": None,
    }
    row.update(overrides)
    return row


def _curated(statement="A lesson.", kind="domain_fact", member_ids=None):
    return CuratedLesson(
        working_statement=statement,
        member_entry_ids=member_ids or [],
        kind=kind,
        novelty="new",
    )


class TestGateContextEntries:
    def test_keeps_confident_unharmed_entries_from_all_sections(self):
        rows = [
            _context_row(section="rules"),
            _context_row(section="goals"),
            _context_row(section="preferences"),
            _context_row(section="lessons_learned"),
        ]
        assert len(gate_context_entries(rows)) == 4

    def test_drops_harmful_and_low_confidence_entries(self):
        rows = [
            _context_row(harmful_count=1),
            _context_row(confidence=0.5),
            _context_row(),
        ]
        gated = gate_context_entries(rows)
        assert len(gated) == 1
        assert gated[0].harmful_count == 0

    def test_ignores_feedback_rows_and_garbage(self):
        rows = [
            {"kind": "feedback", "id": "f1", "raw_text": "x"},
            "not-a-row",
            _context_row(),
        ]
        assert len(gate_context_entries(rows)) == 1


class TestBuildSessionDigest:
    def test_one_line_per_turn_questions_only(self):
        digest = build_session_digest([_qa_row("First?"), _qa_row("Second?")])
        assert digest == "- First?\n- Second?"

    def test_skips_empty_questions_and_collapses_whitespace(self):
        digest = build_session_digest([_qa_row(""), _qa_row("Multi\nline   question?")])
        assert digest == "- Multi line question?"

    def test_truncates_long_questions(self):
        digest = build_session_digest([_qa_row("x" * 500)])
        assert len(digest) < 200


class TestSelectEvidenceExcerpts:
    def test_finds_origin_turn_via_feedback_entry(self):
        entry = _context_row(source_feedback_ids=["fb-1"])
        gated = gate_context_entries([entry])
        origin_qa = _qa_row("Origin question?")
        context_rows = [
            entry,
            {"kind": "feedback", "id": "fb-1", "referenced_qa_ids": [origin_qa["qa_id"]]},
        ]

        excerpts = select_evidence_excerpts(gated, context_rows, [origin_qa, _qa_row("Other?")])

        assert len(excerpts) == 1
        assert "Origin question?" in excerpts[0]

    def test_finds_turns_where_entry_was_served(self):
        entry = _context_row()
        gated = gate_context_entries([entry])
        served_qa = _qa_row("Served turn?", used_session_context_ids=[entry["id"]])

        excerpts = select_evidence_excerpts(gated, [entry], [served_qa, _qa_row("Unrelated?")])

        assert len(excerpts) == 1
        assert "Served turn?" in excerpts[0]

    def test_caps_excerpts_and_truncates_answers(self):
        entry = _context_row()
        gated = gate_context_entries([entry])
        qa_rows = [
            _qa_row(f"Q{i}?", answer="a" * 2000, used_session_context_ids=[entry["id"]])
            for i in range(MAX_EVIDENCE_EXCERPTS + 3)
        ]

        excerpts = select_evidence_excerpts(gated, [entry], qa_rows)

        assert len(excerpts) == MAX_EVIDENCE_EXCERPTS
        assert all(len(excerpt) < 600 for excerpt in excerpts)

    def test_no_matching_turns_returns_empty(self):
        entry = _context_row()
        gated = gate_context_entries([entry])
        assert select_evidence_excerpts(gated, [entry], [_qa_row()]) == []


class TestBuildSourceMessagesByEntry:
    def test_maps_entries_to_their_creating_user_messages(self):
        entry = _context_row(source_feedback_ids=["fb-1", "fb-missing"])
        gated = gate_context_entries([entry])
        context_rows = [
            entry,
            {"kind": "feedback", "id": "fb-1", "raw_text": "Important:\nflashing  wipes data."},
        ]

        messages = build_source_messages_by_entry(gated, context_rows)

        assert messages == {entry["id"]: ["Important: flashing wipes data."]}

    def test_entries_without_known_feedback_are_omitted(self):
        entry = _context_row(source_feedback_ids=[])
        gated = gate_context_entries([entry])
        assert build_source_messages_by_entry(gated, [entry]) == {}

    def test_long_messages_are_truncated(self):
        entry = _context_row(source_feedback_ids=["fb-1"])
        gated = gate_context_entries([entry])
        context_rows = [entry, {"kind": "feedback", "id": "fb-1", "raw_text": "x" * 1000}]

        messages = build_source_messages_by_entry(gated, context_rows)

        assert len(messages[entry["id"]][0]) <= 240


class TestRenderDistilledDocument:
    def test_renders_header_summary_and_kind_sections(self):
        lessons = [
            (
                _curated(kind="domain_fact"),
                DistilledLesson(
                    statement="RoutePulse predicts delivery delays for European freight.",
                    entities=["RoutePulse"],
                    why_learned="Learned while planning the audit trip",
                ),
            ),
            (
                _curated(kind="working_practice"),
                DistilledLesson(statement="Talk to Priya Tan before Mateo Reed."),
            ),
        ]

        document = render_distilled_document(
            session_id="s-1",
            distilled_on="2026-06-11",
            session_summary="A session about audit planning.",
            lessons=lessons,
        )

        assert document.startswith("# Session learnings — 2026-06-11 (session s-1)")
        assert "A session about audit planning." in document
        assert "## What was learned about the domain" in document
        assert "## How to work on this" in document
        assert "(Learned while planning the audit trip.)" in document
        assert "Talk to Priya Tan before Mateo Reed." in document

    def test_empty_kind_section_is_omitted(self):
        lessons = [(_curated(kind="domain_fact"), DistilledLesson(statement="Fact."))]
        document = render_distilled_document(
            session_id="s-1",
            distilled_on="2026-06-11",
            session_summary="Summary.",
            lessons=lessons,
        )
        assert "## How to work on this" not in document

    def test_why_learned_is_optional(self):
        lessons = [(_curated(), DistilledLesson(statement="Plain statement."))]
        document = render_distilled_document(
            session_id="s-1",
            distilled_on="2026-06-11",
            session_summary="Summary.",
            lessons=lessons,
        )
        assert "Plain statement.\n" in document
        assert "()" not in document
