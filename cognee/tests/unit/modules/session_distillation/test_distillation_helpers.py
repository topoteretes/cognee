"""Unit tests for the deterministic parts of session distillation.

The gate, the session digest, the provenance source-message map, and the document
renderer are pure functions, so they are tested directly with fixtures — no LLM, cache,
or vector engine involved.
"""

from uuid import uuid4

from cognee.modules.session_distillation.distill import (
    build_session_digest,
    build_source_messages_by_entry,
    gate_context_entries,
    render_lesson_document,
)
from cognee.modules.session_distillation.models import DistilledLesson


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


class TestRenderLessonDocument:
    def test_renders_standalone_document_with_provenance_header(self):
        document = render_lesson_document(
            DistilledLesson(
                statement="RoutePulse predicts delivery delays for European freight.",
                entities=["RoutePulse"],
                why_learned="Learned while planning the audit trip",
            ),
            session_id="s-1",
            distilled_on="2026-06-11",
        )

        assert document.startswith("# Session learning — 2026-06-11 (session s-1)")
        assert "RoutePulse predicts delivery delays for European freight." in document
        assert "(Learned while planning the audit trip.)" in document

    def test_one_document_holds_exactly_one_lesson(self):
        document = render_lesson_document(
            DistilledLesson(statement="Talk to Priya Tan before Mateo Reed."),
            session_id="s-1",
            distilled_on="2026-06-11",
        )
        # No cross-lesson grouping headings; the doc is a single learning.
        assert "## " not in document
        assert document.count("# Session learning") == 1

    def test_why_learned_is_optional(self):
        document = render_lesson_document(
            DistilledLesson(statement="Plain statement."),
            session_id="s-1",
            distilled_on="2026-06-11",
        )
        assert "Plain statement.\n" in document
        assert "()" not in document
