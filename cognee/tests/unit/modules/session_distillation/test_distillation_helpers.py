"""Unit tests for the deterministic parts of session distillation.

The gate, the timeline batching, and the document renderer are pure functions, so they
are tested directly with fixtures — no LLM, cache, or vector engine involved.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
import cognee.modules.session_distillation.distill as distill_module
from cognee.modules.session_distillation.distill import (
    build_batches,
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


def _context_entry(**overrides):
    return distill_module.coerce_active_context_entries([_context_row(**overrides)])[0]


class TestLoadDistillableSessionInputs:
    @pytest.mark.asyncio
    async def test_loads_qa_and_keeps_confident_unharmed_entries_from_all_sections(
        self, monkeypatch
    ):
        session_manager = SimpleNamespace(
            get_session_context_entries=AsyncMock(
                return_value=[
                    _context_row(section="rules"),
                    _context_row(section="goals"),
                    _context_row(section="preferences"),
                    _context_row(section="lessons_learned"),
                ]
            ),
            get_session=AsyncMock(return_value=[_qa_row(question="What changed?")]),
        )
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: session_manager)

        qa_rows, context_entries = await distill_module._load_distillable_session_inputs(
            SimpleNamespace(user_id="u-1", session_id="s-1")
        )

        assert len(qa_rows) == 1
        assert qa_rows[0]["question"] == "What changed?"
        assert len(context_entries) == 4
        session_manager.get_session_context_entries.assert_awaited_once_with(
            user_id="u-1",
            session_id="s-1",
        )
        session_manager.get_session.assert_awaited_once_with(
            user_id="u-1",
            session_id="s-1",
            formatted=False,
        )

    @pytest.mark.asyncio
    async def test_drops_harmful_low_confidence_feedback_and_garbage(self, monkeypatch):
        session_manager = SimpleNamespace(
            get_session_context_entries=AsyncMock(
                return_value=[
                    _context_row(harmful_count=1),
                    _context_row(confidence=0.5),
                    {"kind": "feedback", "id": "f1", "raw_text": "x"},
                    "not-a-row",
                    _context_row(),
                ]
            ),
            get_session=AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: session_manager)

        _qa_rows, context_entries = await distill_module._load_distillable_session_inputs(
            SimpleNamespace(user_id="u-1", session_id="s-1")
        )

        assert len(context_entries) == 1
        assert context_entries[0].harmful_count == 0


class TestBuildBatches:
    def test_small_session_is_one_batch_with_turns_and_candidates(self):
        context_entries = [_context_entry(content="Flashing wipes calibration.")]
        batches = build_batches([_qa_row("How do I update firmware?")], context_entries)

        assert len(batches) == 1
        assert "User: How do I update firmware?" in batches[0]
        assert f"Candidate {context_entries[0].id}" in batches[0]
        assert "Flashing wipes calibration." in batches[0]

    def test_interleaves_turns_and_candidates_chronologically(self):
        context_entries = [
            _context_entry(content="Earlier candidate.", created_at="2026-06-11T10:00:00")
        ]
        qa = [_qa_row("Later question?", time="2026-06-11T10:00:05")]

        batch = build_batches(qa, context_entries)[0]

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


class TestDistillSessionBoundary:
    @pytest.mark.asyncio
    async def test_requires_dataset(self):
        user = SimpleNamespace(id=uuid4())

        with pytest.raises(CogneeValidationError, match="dataset is required"):
            await distill_module.distill_session("s-1", dataset=None, user=user)

    @pytest.mark.asyncio
    async def test_uses_authorized_write_resolver_for_dataset(self, monkeypatch):
        user = SimpleNamespace(id=uuid4())
        dataset = SimpleNamespace(id=uuid4(), name="team", owner_id=uuid4())
        session_manager = SimpleNamespace(
            get_session_context_entries=AsyncMock(return_value=[]),
            get_session=AsyncMock(return_value=[]),
        )
        calls = []

        async def fake_get_authorized_existing_datasets(datasets, permission_type, resolved_user):
            calls.append((datasets, permission_type, resolved_user))
            return [dataset]

        monkeypatch.setattr(
            distill_module,
            "get_authorized_existing_datasets",
            fake_get_authorized_existing_datasets,
        )
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: session_manager)

        result = await distill_module.distill_session("s-1", dataset="team", user=user)

        assert result.status == "no_gated_entries"
        assert result.dataset_id == str(dataset.id)
        assert calls == [(["team"], "write", user)]
        session_manager.get_session_context_entries.assert_awaited_once_with(
            user_id=str(user.id),
            session_id="s-1",
        )

    @pytest.mark.asyncio
    async def test_rejects_missing_or_unwritable_dataset(self, monkeypatch):
        user = SimpleNamespace(id=uuid4())

        async def fake_get_authorized_existing_datasets(datasets, permission_type, resolved_user):
            assert datasets == ["team"]
            assert permission_type == "write"
            assert resolved_user is user
            return []

        monkeypatch.setattr(
            distill_module,
            "get_authorized_existing_datasets",
            fake_get_authorized_existing_datasets,
        )

        with pytest.raises(CogneeValidationError, match="not found or not writable"):
            await distill_module.distill_session("s-1", dataset="team", user=user)
