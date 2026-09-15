"""Distillation watermark (SDK-593): the curator sees each entry and QA turn once."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import cognee.modules.session_distillation.distill as distill_module
from cognee.infrastructure.session.session_context_models import SessionContextEntry
from cognee.infrastructure.session.session_distillation_watermark import (
    SESSION_DISTILLATION_STATE_ID,
    SESSION_DISTILLATION_STATE_KIND,
    filter_distillable_entries,
    filter_distillable_qa_rows,
    is_newer_than_watermark,
    next_distillation_watermark,
    read_distillation_watermark,
)
from cognee.modules.session_distillation.models import ProposedLesson

T1, T2, T3 = (
    "2026-09-09T10:00:00+00:00",
    "2026-09-09T11:00:00+00:00",
    "2026-09-09T12:00:00+00:00",
)


def _row(created_at, **overrides):
    row = {
        "id": str(uuid4()),
        "section": "lessons_learned",
        "content": "A lesson.",
        "normalized_content": "a lesson.",
        "confidence": 0.9,
        "created_at": created_at,
        "harmful_count": 0,
        "kind": "context",
    }
    row.update(overrides)
    return row


def _entry(created_at, **overrides):
    return SessionContextEntry.model_validate(_row(created_at, **overrides))


def _qa(time):
    return {"question": "q", "answer": "a", "time": time}


def _watermark_row(through):
    return {
        "id": SESSION_DISTILLATION_STATE_ID,
        "kind": SESSION_DISTILLATION_STATE_KIND,
        "distilled_through": through,
    }


class FakeSessionManager:
    def __init__(self, context_rows, qa_rows):
        self.context = list(context_rows)
        self.qa = list(qa_rows)
        self.get_session_context_entries = AsyncMock(side_effect=self._ctx)
        self.get_session = AsyncMock(side_effect=self._qa)

    async def _ctx(self, *, user_id, session_id):
        return list(self.context)

    async def _qa(self, *, user_id, session_id, formatted=False):
        return list(self.qa)

    async def update_session_context_entry(self, *, user_id, session_id, entry_id, merge):
        for row in self.context:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, session_id, entry_dump):
        self.context.append(dict(entry_dump))
        return True

    def watermark(self):
        return read_distillation_watermark(self.context)


class TestPureHelpers:
    def test_missing_or_malformed_watermark_reads_as_empty(self):
        assert read_distillation_watermark([]) == ""
        assert read_distillation_watermark([_row(T1)]) == ""
        assert read_distillation_watermark([_watermark_row(None)]) == ""
        assert read_distillation_watermark([_watermark_row(T2)]) == T2

    def test_empty_watermark_matches_everything(self):
        assert is_newer_than_watermark(T1, "") is True
        assert is_newer_than_watermark("", "") is True

    def test_strictly_newer_only(self):
        assert is_newer_than_watermark(T2, T1) is True
        assert is_newer_than_watermark(T1, T1) is False
        assert is_newer_than_watermark(None, T1) is False

    def test_entries_filter_by_gate_and_watermark(self):
        entries = [
            _entry(T1),
            _entry(T2),
            _entry(T3, confidence=0.1),
            _entry(T3, harmful_count=1),
            _entry(T3),
        ]
        kept = filter_distillable_entries(entries, T1)
        assert [entry.created_at for entry in kept] == [T2, T3]

    def test_qa_rows_filter_by_watermark(self):
        assert filter_distillable_qa_rows([_qa(T1), _qa(T2), {"question": "no time"}], T1) == [
            _qa(T2)
        ]

    def test_next_watermark_is_the_newest_consumed_stamp_and_never_regresses(self):
        assert next_distillation_watermark([_qa(T1)], [_entry(T2)], "") == T2
        assert next_distillation_watermark([_qa(T3)], [_entry(T2)], "") == T3
        assert next_distillation_watermark([], [], T3) == T3
        assert next_distillation_watermark([_qa(T1)], [], T3) == T3


class TestLoadAndAdvance:
    @pytest.mark.asyncio
    async def test_load_cuts_both_lists_at_the_watermark(self, monkeypatch):
        manager = FakeSessionManager([_row(T1), _row(T2), _watermark_row(T1)], [_qa(T1), _qa(T2)])
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)

        qa_rows, entries = await distill_module.load_distillable_session_inputs(
            SimpleNamespace(user_id="u", session_id="s")
        )

        assert [row["time"] for row in qa_rows] == [T2]
        assert [entry.created_at for entry in entries] == [T2]

    @pytest.mark.asyncio
    async def test_distill_session_advances_on_no_proposed_lessons(self, monkeypatch):
        """Entries the curator judged and dropped are not re-judged next time."""
        manager = FakeSessionManager([_row(T1), _row(T2)], [_qa(T1)])
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)
        monkeypatch.setattr(
            distill_module,
            "resolve_distillation_scope",
            AsyncMock(
                return_value=SimpleNamespace(
                    user_id="u",
                    session_id="s",
                    dataset_id="d",
                    result=lambda status, documents=None: SimpleNamespace(
                        status=status, documents=documents or []
                    ),
                )
            ),
        )
        monkeypatch.setattr(distill_module, "propose_lessons", AsyncMock(return_value=[]))

        result = await distill_module.distill_session("s", dataset="d", user=None)

        assert result.status == "no_proposed_lessons"
        assert manager.watermark() == T2

        # Second run: nothing above the watermark -> no curator call at all.
        distill_module.propose_lessons.reset_mock()
        result = await distill_module.distill_session("s", dataset="d", user=None)
        assert result.status == "no_gated_entries"
        distill_module.propose_lessons.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_distill_session_advances_only_after_a_successful_publish(self, monkeypatch):
        manager = FakeSessionManager([_row(T2)], [_qa(T1)])
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)
        scope = SimpleNamespace(
            user_id="u",
            session_id="s",
            dataset_id="d",
            result=lambda status, documents=None: SimpleNamespace(
                status=status, documents=documents or []
            ),
        )
        monkeypatch.setattr(
            distill_module, "resolve_distillation_scope", AsyncMock(return_value=scope)
        )
        proposed = [ProposedLesson(working_statement="x", member_entry_ids=[])]
        monkeypatch.setattr(distill_module, "propose_lessons", AsyncMock(return_value=proposed))
        monkeypatch.setattr(
            distill_module,
            "accept_proposed_lessons",
            AsyncMock(return_value=[SimpleNamespace(statement="s", why_learned="w", accept=True)]),
        )
        monkeypatch.setattr(
            distill_module,
            "publish_distilled_lessons",
            AsyncMock(side_effect=RuntimeError("cognify down")),
        )

        with pytest.raises(RuntimeError):
            await distill_module.distill_session("s", dataset="d", user=None)
        assert manager.watermark() == ""  # retry the same window next time

        distill_module.publish_distilled_lessons = AsyncMock(return_value=["doc"])
        result = await distill_module.distill_session("s", dataset="d", user=None)

        assert result.status == "completed"
        assert manager.watermark() == T2

    @pytest.mark.asyncio
    async def test_a_newer_entry_is_distilled_on_the_next_run(self, monkeypatch):
        manager = FakeSessionManager([_row(T1), _watermark_row(T1)], [])
        monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)
        scope = SimpleNamespace(
            user_id="u",
            session_id="s",
            dataset_id="d",
            result=lambda status, documents=None: SimpleNamespace(
                status=status, documents=documents or []
            ),
        )
        monkeypatch.setattr(
            distill_module, "resolve_distillation_scope", AsyncMock(return_value=scope)
        )
        propose = AsyncMock(return_value=[])
        monkeypatch.setattr(distill_module, "propose_lessons", propose)

        assert (await distill_module.distill_session("s", dataset="d", user=None)).status == (
            "no_gated_entries"
        )
        manager.context.append(_row(T3))

        assert (await distill_module.distill_session("s", dataset="d", user=None)).status == (
            "no_proposed_lessons"
        )
        (_qa_rows, entries), _ = propose.await_args
        assert [entry.created_at for entry in entries] == [T3]
        assert manager.watermark() == T3
