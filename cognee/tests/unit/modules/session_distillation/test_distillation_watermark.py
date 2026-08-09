"""Distillation watermark: a re-run over an unchanged session must cost zero LLM calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.session_distillation import distill as distill_module
from cognee.modules.session_distillation.distill import (
    SessionDistillationScope,
    _dedupe_by_statement,
    _distillation_state_id,
    distill_session,
)
from cognee.modules.session_distillation.models import WrittenLesson


def _scope():
    return SessionDistillationScope(
        session_id="s1",
        user=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=uuid4(), owner_id=uuid4()),
    )


class _StateStore:
    """In-memory stand-in for the session-context state rows."""

    def __init__(self):
        self.rows = {}

    def session_manager(self):
        sm = SimpleNamespace()
        sm.get_session_context_entries = AsyncMock(
            side_effect=lambda **_kw: list(self.rows.values())
        )

        async def update(*, user_id, session_id, entry_id, merge):
            if entry_id in self.rows:
                self.rows[entry_id] = {**self.rows[entry_id], **merge}
                return True
            return False

        async def create(*, user_id, session_id, entry_dump):
            self.rows[entry_dump["id"]] = dict(entry_dump)

        sm.update_session_context_entry = AsyncMock(side_effect=update)
        sm.create_session_context_entry = AsyncMock(side_effect=create)
        return sm


def _entry(entry_id):
    return SimpleNamespace(id=entry_id, harmful_count=0, confidence=0.9, content="c")


@pytest.mark.asyncio
async def test_second_run_on_unchanged_session_skips_all_llm_work():
    scope = _scope()
    store = _StateStore()
    proposer = AsyncMock(return_value=[SimpleNamespace(working_statement="w", member_entry_ids=[])])
    acceptor = AsyncMock(return_value=[WrittenLesson(accept=True, statement="lesson")])
    publisher = AsyncMock(return_value=["doc"])

    with (
        patch.object(
            distill_module, "resolve_distillation_scope", new=AsyncMock(return_value=scope)
        ),
        patch.object(
            distill_module,
            "load_distillable_session_inputs",
            new=AsyncMock(return_value=([], [_entry("e1")])),
        ),
        patch.object(distill_module, "get_session_manager", new=store.session_manager),
        patch.object(distill_module, "propose_lessons", new=proposer),
        patch.object(distill_module, "accept_proposed_lessons", new=acceptor),
        patch.object(distill_module, "publish_distilled_lessons", new=publisher),
    ):
        first = await distill_session("s1", dataset="docs")
        second = await distill_session("s1", dataset="docs")

    assert first.status == "completed"
    assert second.status == "no_new_entries"
    proposer.assert_awaited_once()
    acceptor.assert_awaited_once()
    publisher.assert_awaited_once()
    state = store.rows[_distillation_state_id(scope.dataset_id)]
    assert state["processed_entry_ids"] == ["e1"]


@pytest.mark.asyncio
async def test_new_gated_entry_reopens_distillation():
    scope = _scope()
    store = _StateStore()
    entries = [[_entry("e1")], [_entry("e1"), _entry("e2")]]
    loader = AsyncMock(side_effect=[([], batch) for batch in entries])
    proposer = AsyncMock(return_value=[])

    with (
        patch.object(
            distill_module, "resolve_distillation_scope", new=AsyncMock(return_value=scope)
        ),
        patch.object(distill_module, "load_distillable_session_inputs", new=loader),
        patch.object(distill_module, "get_session_manager", new=store.session_manager),
        patch.object(distill_module, "propose_lessons", new=proposer),
    ):
        first = await distill_session("s1", dataset="docs")
        second = await distill_session("s1", dataset="docs")

    assert first.status == "no_proposed_lessons"
    assert second.status == "no_proposed_lessons"
    assert proposer.await_count == 2  # e2 arrived, so the second run curates again
    state = store.rows[_distillation_state_id(scope.dataset_id)]
    assert state["processed_entry_ids"] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_failed_run_does_not_advance_watermark():
    scope = _scope()
    store = _StateStore()

    with (
        patch.object(
            distill_module, "resolve_distillation_scope", new=AsyncMock(return_value=scope)
        ),
        patch.object(
            distill_module,
            "load_distillable_session_inputs",
            new=AsyncMock(return_value=([], [_entry("e1")])),
        ),
        patch.object(distill_module, "get_session_manager", new=store.session_manager),
        patch.object(
            distill_module, "propose_lessons", new=AsyncMock(side_effect=RuntimeError("llm down"))
        ),
    ):
        with pytest.raises(RuntimeError):
            await distill_session("s1", dataset="docs")

    assert _distillation_state_id(scope.dataset_id) not in store.rows


def test_dedupe_by_statement_normalizes_text():
    lessons = [
        SimpleNamespace(statement="Use  UV for installs."),
        SimpleNamespace(statement="use uv for installs."),
        SimpleNamespace(statement="Different lesson."),
        SimpleNamespace(statement="   "),
    ]
    unique = _dedupe_by_statement(lessons, lambda lesson: lesson.statement)
    assert [lesson.statement for lesson in unique] == [
        "Use  UV for installs.",
        "Different lesson.",
    ]
