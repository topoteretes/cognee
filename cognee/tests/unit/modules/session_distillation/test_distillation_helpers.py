"""Unit tests for the deterministic parts of session distillation.

The gate, the timeline batching, and the document renderer are pure functions, so they
are tested directly with fixtures — no LLM, cache, or vector engine involved.
"""

from uuid import uuid4

from cognee.modules.session_distillation.distill import (
    build_batches,
    gate_context_entries,
    render_lesson_document,
)
from cognee.modules.session_distillation.models import BATCH_CHAR_BUDGET, WrittenLesson


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


def _qa_row(question="What?", answer="That.", time="2026-06-11T10:00:00", **overrides):
    row = {
        "time": time,
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


class TestBuildBatches:
    def test_small_session_is_one_batch_with_turns_and_candidates(self):
        gated = gate_context_entries([_context_row(content="Flashing wipes calibration.")])
        batches = build_batches([_qa_row("How do I update firmware?")], gated)

        assert len(batches) == 1
        assert "User: How do I update firmware?" in batches[0]
        assert f"Candidate {gated[0].id}" in batches[0]
        assert "Flashing wipes calibration." in batches[0]

    def test_interleaves_turns_and_candidates_chronologically(self):
        gated = gate_context_entries(
            [_context_row(content="Earlier candidate.", created_at="2026-06-11T10:00:00")]
        )
        qa = [_qa_row("Later question?", time="2026-06-11T10:00:05")]

        batch = build_batches(qa, gated)[0]

        assert batch.index("Earlier candidate.") < batch.index("Later question?")

    def test_splits_when_over_char_budget(self):
        # Context-entry content is capped at 280 chars, so size the timeline with long
        # questions (uncapped) instead. Three ~9k blocks pack into three batches.
        big_question = "x" * 9000
        qa = [_qa_row(question=big_question, time=f"2026-06-11T10:00:0{i}") for i in range(3)]
        batches = build_batches(qa, [])

        assert len(batches) == 3

    def test_oversized_single_block_gets_its_own_batch(self):
        huge_question = "y" * (BATCH_CHAR_BUDGET * 2)
        batches = build_batches([_qa_row(question=huge_question)], [])

        assert len(batches) == 1
        assert huge_question in batches[0]


class TestRenderLessonDocument:
    def test_renders_standalone_document_with_provenance_header(self):
        document = render_lesson_document(
            WrittenLesson(
                accept=True,
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
            WrittenLesson(accept=True, statement="Talk to Priya Tan before Mateo Reed."),
            session_id="s-1",
            distilled_on="2026-06-11",
        )
        assert "## " not in document
        assert document.count("# Session learning") == 1

    def test_why_learned_is_optional(self):
        document = render_lesson_document(
            WrittenLesson(accept=True, statement="Plain statement."),
            session_id="s-1",
            distilled_on="2026-06-11",
        )
        assert "Plain statement.\n" in document
        assert "()" not in document
