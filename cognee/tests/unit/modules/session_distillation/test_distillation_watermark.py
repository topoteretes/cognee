"""Distillation watermark (improve stage 5).

A per-(session, dataset) watermark records the gated context entries a
completed run already covered. When nothing new is gated, ``distill_session``
returns ``no_new_entries`` and makes zero LLM calls. The watermark advances
only once a run finished (lessons published, or the LLM found nothing durable)
and never when the run raised. Lesson documents carry no run date so a
re-accepted lesson hashes identically at add().
"""

import re
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.session_distillation.distill as distill_module
from cognee.infrastructure.session.session_persist_watermark import (
    get_distilled_entry_ids,
    save_distilled_entry_ids,
)
from cognee.modules.session_distillation.models import ProposedLesson, WrittenLesson

USER = SimpleNamespace(id=uuid4())
DATASET_A = SimpleNamespace(id=uuid4(), name="team-a", owner_id=USER.id)
DATASET_B = SimpleNamespace(id=uuid4(), name="team-b", owner_id=USER.id)


def _context_row(content="A lesson.", **overrides):
    row = {
        "id": str(uuid4()),
        "section": "lessons_learned",
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


class FakeSessionManager:
    def __init__(self, rows=None):
        self.store: list[dict] = [dict(row) for row in (rows or [])]

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.store)

    async def get_session(self, **kwargs):
        return []

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.store:
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.store.append(dict(entry_dump))
        return True


@pytest.fixture
def datasets(monkeypatch):
    by_name = {DATASET_A.name: DATASET_A, DATASET_B.name: DATASET_B}

    async def fake_get_authorized_existing_datasets(requested, permission_type, resolved_user):
        return [by_name[requested[0]]]

    monkeypatch.setattr(
        distill_module, "get_authorized_existing_datasets", fake_get_authorized_existing_datasets
    )
    return by_name


def _forbid_llm(monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("distillation must make no LLM call when nothing new is gated")

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", unexpected)
    monkeypatch.setattr(distill_module, "propose_lessons", unexpected)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", unexpected)


def _install_manager(monkeypatch, manager):
    monkeypatch.setattr(distill_module, "get_session_manager", lambda: manager)


@pytest.mark.asyncio
async def test_all_gated_entries_already_distilled_returns_no_new_entries(monkeypatch, datasets):
    rows = [_context_row("Lesson one."), _context_row("Lesson two.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(
        manager, str(USER.id), "s-1", str(DATASET_A.id), [row["id"] for row in rows]
    )
    _forbid_llm(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_new_entries"
    assert result.documents == []
    assert result.dataset_id == str(DATASET_A.id)


@pytest.mark.asyncio
async def test_watermark_is_scoped_to_the_target_dataset(monkeypatch, datasets):
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id), [rows[0]["id"]])

    proposed = []

    async def fake_propose(qa_rows, context_entries):
        proposed.append([entry.id for entry in context_entries])
        return []

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    result = await distill_module.distill_session("s-1", dataset=DATASET_B.name, user=USER)

    # Dataset B has never seen this session: the run happens.
    assert result.status == "no_proposed_lessons"
    assert proposed == [[rows[0]["id"]]]


@pytest.mark.asyncio
async def test_no_gated_entries_still_wins_over_the_watermark(monkeypatch, datasets):
    manager = FakeSessionManager([_context_row(confidence=0.2)])
    _install_manager(monkeypatch, manager)
    _forbid_llm(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_gated_entries"
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()


@pytest.mark.asyncio
async def test_new_gated_entry_after_the_watermark_triggers_a_run(monkeypatch, datasets):
    old_row = _context_row("Old lesson.")
    manager = FakeSessionManager([old_row])
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id), [old_row["id"]])

    new_row = _context_row("New lesson.")
    manager.store.append(dict(new_row))

    seen = []

    async def fake_propose(qa_rows, context_entries):
        seen.append(sorted(entry.id for entry in context_entries))
        return []

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_proposed_lessons"
    # The curator still sees the whole gated timeline; the watermark only decides
    # whether the run happens at all.
    assert seen == [sorted([old_row["id"], new_row["id"]])]
    # Both entries are now covered.
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        old_row["id"],
        new_row["id"],
    }


@pytest.mark.asyncio
async def test_no_proposed_lessons_advances_watermark_so_next_run_is_free(monkeypatch, datasets):
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    calls = 0

    async def fake_propose(qa_rows, context_entries):
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    first = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)
    second = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert first.status == "no_proposed_lessons"
    assert second.status == "no_new_entries"
    assert calls == 1


@pytest.mark.asyncio
async def test_no_accepted_lessons_advances_watermark(monkeypatch, datasets):
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        return [ProposedLesson(working_statement="Something.")]

    async def fake_accept(scope, proposed, context_entries):
        return []

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_accepted_lessons"
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        rows[0]["id"]
    }


@pytest.mark.asyncio
async def test_completed_run_advances_watermark_after_publish(monkeypatch, datasets):
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    order = []

    async def fake_propose(qa_rows, context_entries):
        return [ProposedLesson(working_statement="Something.")]

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")]

    async def fake_publish(scope, accepted):
        order.append(("publish", len(manager.store)))
        return ["doc"]

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", fake_publish)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "completed"
    assert result.documents == ["doc"]
    # Publish ran before the watermark row existed (only the gated row was stored).
    assert order == [("publish", 1)]
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        rows[0]["id"]
    }


@pytest.mark.asyncio
async def test_failed_publish_leaves_watermark_untouched(monkeypatch, datasets):
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        return [ProposedLesson(working_statement="Something.")]

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")]

    async def failing_publish(scope, accepted):
        raise RuntimeError("cognify exploded")

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", failing_publish)

    with pytest.raises(RuntimeError, match="cognify exploded"):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()
    # The retried window is the same one: the next run is not skipped.
    assert (
        len(
            await distill_module.select_undistilled_entries(
                SimpleNamespace(
                    user_id=str(USER.id), session_id="s-1", dataset_id=str(DATASET_A.id)
                ),
                distill_module.coerce_active_context_entries(manager.store),
            )
        )
        == 1
    )


def test_lesson_documents_carry_no_run_date():
    lesson = WrittenLesson(
        accept=True,
        statement="RoutePulse predicts delivery delays.",
        why_learned="Learned during the audit",
    )
    document = distill_module.render_lesson_document(lesson, session_id="s-1")

    assert document.startswith("# Session learning (session s-1)")
    assert re.search(r"\d{4}-\d{2}-\d{2}", document) is None
    # Identical lessons render byte-identically, so add()'s content hash dedups them.
    assert document == distill_module.render_lesson_document(lesson, session_id="s-1")
