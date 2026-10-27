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
from cognee.modules.session_distillation import models as distill_models
from cognee.modules.session_distillation.models import (
    CuratorBatchOutput,
    ProposedLesson,
    WrittenLesson,
)

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
    def __init__(self, rows=None, qa_rows=None):
        self.store: list[dict] = [dict(row) for row in (rows or [])]
        self.qa_rows: list[dict] = [dict(row) for row in (qa_rows or [])]
        self.qa_reads: list[dict] = []

    async def get_session_context_entries(self, *, user_id, session_id=None):
        return list(self.store)

    async def get_session(self, *, since=None, lead_in=0, **kwargs):
        """Same contract as the cache interface's get_qa_entries_since."""
        from datetime import datetime, timezone

        self.qa_reads.append({"since": since, "lead_in": lead_in})
        if since is None:
            return list(self.qa_rows)
        first = len(self.qa_rows)
        for index, row in enumerate(self.qa_rows):
            stored = datetime.fromisoformat(row["time"])
            if stored.tzinfo is None:
                stored = stored.replace(tzinfo=timezone.utc)
            if stored >= since:
                first = index
                break
        return list(self.qa_rows[max(first - lead_in, 0) :])

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


def _curator_run(context_entries, proposed=None, failed_batches=0, curated=None):
    """Stand-in for propose_lessons: by default every batch completed."""
    entries = context_entries if curated is None else curated
    return distill_module.CuratorRun(
        proposed=list(proposed or []),
        failed_batches=failed_batches,
        curated_entry_ids={entry.id for entry in entries},
    )


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
        return _curator_run(context_entries)

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
        return _curator_run(context_entries)

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_proposed_lessons"
    # The curator sees only the entry the watermark does not cover yet.
    assert seen == [[new_row["id"]]]
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
        return _curator_run(context_entries)

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
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def fake_accept(scope, proposed, context_entries):
        return [], 0

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
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")], 0

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
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")], 0

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


@pytest.mark.asyncio
async def test_failed_curator_calls_leave_the_watermark_untouched(monkeypatch, datasets):
    """An LLM outage is not "nothing durable": sealing the entries would mark a
    finished session's learnings distilled forever without a single call running."""
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        return _curator_run(context_entries, failed_batches=2, curated=[])

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    with pytest.raises(distill_module.DistillationCallsFailedError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()


@pytest.mark.asyncio
async def test_partial_failure_publishes_survivors_and_seals_only_completed_batches(
    monkeypatch, datasets
):
    done_row = _context_row("Lesson one.")
    lost_row = _context_row("Lesson two.")
    manager = FakeSessionManager([done_row, lost_row])
    _install_manager(monkeypatch, manager)
    published = []

    async def fake_propose(qa_rows, context_entries):
        done = [entry for entry in context_entries if entry.id == done_row["id"]]
        return _curator_run(
            context_entries,
            [ProposedLesson(working_statement="Something.")],
            failed_batches=1,
            curated=done,
        )

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")], 0

    async def fake_publish(scope, accepted):
        published.append(len(accepted))
        return ["doc"]

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", fake_publish)

    with pytest.raises(distill_module.DistillationCallsFailedError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    # Surviving lessons were published (date-free documents dedup on the retry).
    # The completed batch is sealed; only the lost batch's entry comes back.
    assert published == [1]
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        done_row["id"]
    }

    seen = []

    async def fake_propose_again(qa_rows, context_entries):
        seen.extend(entry.id for entry in context_entries)
        return _curator_run(context_entries)

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose_again)
    second = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert second.status == "no_proposed_lessons"
    assert seen == [lost_row["id"]]


@pytest.mark.asyncio
async def test_partial_failure_seals_nothing_when_publish_fails(monkeypatch, datasets):
    """Completed batches are sealed only once their lessons have landed, so a
    publish failure on the partial path leaves everything for the retry."""
    done_row = _context_row("Lesson one.")
    lost_row = _context_row("Lesson two.")
    manager = FakeSessionManager([done_row, lost_row])
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        done = [entry for entry in context_entries if entry.id == done_row["id"]]
        return _curator_run(
            context_entries,
            [ProposedLesson(working_statement="Something.")],
            failed_batches=1,
            curated=done,
        )

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")], 0

    async def failing_publish(scope, accepted):
        raise RuntimeError("cognify down")

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", failing_publish)

    with pytest.raises(RuntimeError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()


@pytest.mark.asyncio
async def test_exhausted_budget_in_the_writer_phase_seals_nothing(monkeypatch, datasets):
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def broke_accept(scope, proposed, context_entries):
        raise LLMPaymentRequiredError("402")

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", broke_accept)

    with pytest.raises(LLMPaymentRequiredError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()


@pytest.mark.asyncio
async def test_accept_proposed_lessons_raises_on_exhausted_budget_and_skips_the_rest(
    monkeypatch, datasets
):
    from contextlib import asynccontextmanager

    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    @asynccontextmanager
    async def no_context(*args, **kwargs):
        yield

    async def no_engine():
        return None

    calls = []

    async def broke(vector_engine, lesson, entries_by_id):
        calls.append(lesson.working_statement)
        raise LLMPaymentRequiredError("402")

    monkeypatch.setattr(distill_module, "set_database_global_context_variables", no_context)
    monkeypatch.setattr(distill_module, "get_vector_engine_async", no_engine)
    monkeypatch.setattr(distill_module, "evaluate_proposed_lesson", broke)
    scope = SimpleNamespace(session_id="s-1", dataset=DATASET_A)
    proposed = [ProposedLesson(working_statement=f"Lesson {index}.") for index in range(40)]

    with pytest.raises(LLMPaymentRequiredError):
        await distill_module.accept_proposed_lessons(scope, proposed, [])

    assert len(calls) <= distill_models.WRITER_CONCURRENCY


@pytest.mark.asyncio
async def test_one_failed_curator_call_leaves_exactly_its_batch_pending(monkeypatch, datasets):
    """End to end through the real batching: of 10 curator calls one fails, and the
    next run sees only the entries that were in that call's batch."""
    rows = [
        _context_row(f"Note {index}.", created_at=f"2026-06-11T10:{index:02d}:00+00:00")
        for index in range(60)
    ]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    inputs: list[str] = []

    async def flaky(*, text_input, system_prompt, response_model):
        inputs.append(text_input)
        if len(inputs) == 4:
            raise RuntimeError("transient")
        return CuratorBatchOutput(lessons=[])

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", flaky)
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda name: "PROMPT")

    with pytest.raises(distill_module.DistillationCallsFailedError, match="1 curator batch"):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    failed_text = inputs[3]
    lost = {row["id"] for row in rows if f"Candidate {row['id']}" in failed_text}
    assert len(lost) == distill_models.CURATOR_BLOCKS_PER_BATCH
    sealed = await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id))
    assert sealed == {row["id"] for row in rows} - lost

    seen: list[str] = []

    async def fine(*, text_input, system_prompt, response_model):
        seen.append(text_input)
        return CuratorBatchOutput(lessons=[])

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", fine)
    second = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert second.status == "no_proposed_lessons"
    assert len(seen) == 1
    assert {row["id"] for row in rows if f"Candidate {row['id']}" in seen[0]} == lost


@pytest.mark.asyncio
async def test_partial_seal_keeps_earlier_seals(monkeypatch, datasets):
    """The watermark is replaced on each write, so a partial seal must carry the
    ids sealed by earlier runs or they would come back as new."""
    old_row = _context_row("Old lesson.")
    done_row = _context_row("Lesson one.")
    lost_row = _context_row("Lesson two.")
    manager = FakeSessionManager([old_row, done_row, lost_row])
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id), [old_row["id"]])

    async def fake_propose(qa_rows, context_entries):
        done = [entry for entry in context_entries if entry.id == done_row["id"]]
        return _curator_run(context_entries, failed_batches=1, curated=done)

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)

    with pytest.raises(distill_module.DistillationCallsFailedError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        old_row["id"],
        done_row["id"],
    }


@pytest.mark.asyncio
async def test_failed_writer_calls_seal_the_curated_entries(monkeypatch, datasets):
    """The curator processed the entry, so it is sealed; the lesson the writer lost
    is not retried, and the run still raises so the stage reports the loss."""
    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)

    async def fake_propose(qa_rows, context_entries):
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def fake_accept(scope, proposed, context_entries):
        return [], 1  # the writer call failed, nothing was rejected

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)

    with pytest.raises(distill_module.DistillationCallsFailedError, match="writer call"):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        rows[0]["id"]
    }


@pytest.mark.asyncio
async def test_exhausted_budget_stops_the_run_and_seals_nothing(monkeypatch, datasets):
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError

    rows = [_context_row(f"Note {index}.") for index in range(60)]  # 10 batches
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    calls = []

    async def broke(*, text_input, system_prompt, response_model):
        calls.append(text_input)
        raise LLMPaymentRequiredError("402")

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", broke)
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda name: "PROMPT")

    with pytest.raises(LLMPaymentRequiredError):
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    # Only the calls already in flight hit the provider; the rest were skipped.
    assert len(calls) <= distill_models.CURATOR_CONCURRENCY
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == set()


@pytest.mark.asyncio
async def test_curate_batch_reports_failure_as_none(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", boom)
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda name: "PROMPT")

    assert await distill_module.curate_batch("batch text") is None

    run = await distill_module.propose_lessons([{"question": "q", "answer": "a"}], [])
    assert run.proposed == []
    assert run.failed_batches >= 1


@pytest.mark.asyncio
async def test_idle_run_never_reads_the_qa_history(monkeypatch, datasets):
    """The default debounce runs distillation after every remember(session_id=...),
    so the no_new_entries exit must stay cheap: no QA-history load, no dump."""
    rows = [_context_row("Lesson one.")]

    class CountingManager(FakeSessionManager):
        qa_reads = 0

        async def get_session(self, **kwargs):
            type(self).qa_reads += 1
            return []

    manager = CountingManager(rows)
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id), [rows[0]["id"]])
    _forbid_llm(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_new_entries"
    assert CountingManager.qa_reads == 0


@pytest.mark.asyncio
async def test_turns_before_the_one_that_produced_the_earliest_new_entry_stay_home(
    monkeypatch, datasets
):
    """Turns carry no watermark. The entry is written while the next turn is answered,
    before that turn's QA row, and it judges the turn before; so the newest turn
    before the earliest new entry is sent, older ones are not."""
    from datetime import datetime, timezone

    old_rows = [_context_row("Old lesson.", created_at="2026-06-11T09:00:00.200000+00:00")]
    new_rows = [_context_row("New lesson.", created_at="2026-06-11T12:05:00.200000+00:00")]
    turns = [
        {"time": "2026-06-11T08:00:00.100000+00:00", "question": "Early turn?", "answer": "A."},
        {"time": "2026-06-11T09:00:00.900000+00:00", "question": "Old turn?", "answer": "B."},
        {"time": "2026-06-11T12:05:00.900000+00:00", "question": "No, metric?", "answer": "C."},
        {"time": "2026-06-11T13:00:00.900000+00:00", "question": "Later turn?", "answer": "D."},
    ]

    manager = FakeSessionManager(old_rows + new_rows, qa_rows=turns)
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(
        manager, str(USER.id), "s-1", str(DATASET_A.id), [old_rows[0]["id"]]
    )
    curator_inputs = _record_curator_calls(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_proposed_lessons"
    curated = "\n".join(curator_inputs)
    assert "Early turn?" not in curated
    # "Old turn?" (09:00:00.9) is the turn the new entry (12:05:00.2) was written
    # about: its answer B is what the user corrected, so it goes.
    assert "Old turn?" in curated
    assert "No, metric?" in curated
    assert "Later turn?" in curated
    assert "New lesson." in curated
    # The bound is pushed down to the cache: one read, from the new entry, one turn of lead-in.
    assert manager.qa_reads == [
        {"since": datetime(2026, 6, 11, 12, 5, 0, 200000, tzinfo=timezone.utc), "lead_in": 1}
    ]


def _record_curator_calls(monkeypatch, lessons_per_call=0):
    curator_inputs: list[str] = []

    async def fake_llm(*, text_input, system_prompt, response_model):
        assert response_model is CuratorBatchOutput
        curator_inputs.append(text_input)
        return CuratorBatchOutput(
            lessons=[
                ProposedLesson(working_statement=f"Lesson {len(curator_inputs)}.{index}")
                for index in range(lessons_per_call)
            ]
        )

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", fake_llm)
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda name: "PROMPT")
    return curator_inputs


@pytest.mark.asyncio
async def test_curator_sees_only_entries_not_yet_distilled(monkeypatch, datasets):
    old_rows = [_context_row(f"Already distilled {index}.") for index in range(3)]
    new_rows = [_context_row(f"Fresh lesson {index}.") for index in range(2)]
    manager = FakeSessionManager(old_rows + new_rows)
    _install_manager(monkeypatch, manager)
    await save_distilled_entry_ids(
        manager, str(USER.id), "s-1", str(DATASET_A.id), [row["id"] for row in old_rows]
    )
    curator_inputs = _record_curator_calls(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_proposed_lessons"
    curated = "\n".join(curator_inputs)
    assert "Already distilled" not in curated
    assert all(row["content"] in curated for row in new_rows)


def _set_distill_cap(monkeypatch, max_batches: int) -> None:
    from cognee.modules.improve.config import ImproveConfig

    config = ImproveConfig(_env_file=None, distill_max_batches_per_run=max_batches)
    monkeypatch.setattr(distill_module, "get_improve_config", lambda: config)


@pytest.mark.asyncio
async def test_uncapped_run_curates_every_new_entry(monkeypatch, datasets):
    """Default: no cap. Every new gated entry reaches the curator, so sealing them all
    afterwards loses nothing."""
    _set_distill_cap(monkeypatch, 0)
    rows = [_context_row(f"Note {index}.", confidence=0.8) for index in range(400)]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    curator_inputs = _record_curator_calls(monkeypatch)

    result = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert result.status == "no_proposed_lessons"
    assert result.uncurated_entries == 0
    assert len(curator_inputs) == -(-len(rows) // distill_models.CURATOR_BLOCKS_PER_BATCH)
    curated = "\n".join(curator_inputs)
    assert all(row["content"] in curated for row in rows)
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        row["id"] for row in rows
    }


@pytest.mark.asyncio
async def test_capped_run_curates_the_most_relevant_and_seals_the_rest(monkeypatch, datasets):
    """With IMPROVE_DISTILL_MAX_BATCHES_PER_RUN set, a session that gates thousands of
    entries fans out into at most that many curator calls. The overflow is sealed,
    reported on the result, and never becomes a backlog."""
    cap = 20
    _set_distill_cap(monkeypatch, cap)
    blocks_per_run = cap * distill_models.CURATOR_BLOCKS_PER_BATCH
    sure_rows = [_context_row(f"High-confidence {index}.", confidence=0.99) for index in range(5)]
    bulk_rows = [
        _context_row(f"Subagent note {index}.", confidence=0.8)
        for index in range(blocks_per_run * 3)
    ]
    manager = FakeSessionManager(bulk_rows + sure_rows)
    _install_manager(monkeypatch, manager)
    curator_inputs = _record_curator_calls(monkeypatch, lessons_per_call=2)
    published: list[int] = []

    async def accept_all(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement=p.working_statement) for p in proposed], 0

    async def fake_publish(scope, accepted):
        published.append(len(accepted))
        return [lesson.statement for lesson in accepted]

    monkeypatch.setattr(distill_module, "accept_proposed_lessons", accept_all)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", fake_publish)

    first = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert first.status == "completed"
    assert first.uncurated_entries == len(bulk_rows) + len(sure_rows) - blocks_per_run
    assert 0 < len(curator_inputs) <= cap
    assert sum(published) <= cap * 2
    curated = "\n".join(curator_inputs)
    assert all(row["content"] in curated for row in sure_rows)
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        row["id"] for row in bulk_rows + sure_rows
    }

    _forbid_llm(monkeypatch)
    second = await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)
    assert second.status == "no_new_entries"


@pytest.mark.asyncio
async def test_a_failed_step_after_the_lessons_cognify_is_raised_after_the_watermark(
    monkeypatch, datasets
):
    """The lessons landed; only the step after their cognify failed. The watermark
    still moves, so the curator and writer calls are not repeated on the next
    improve, and then the error is raised."""
    from cognee.modules.pipelines.exceptions import AfterRunHookError

    rows = [_context_row("Lesson one.")]
    manager = FakeSessionManager(rows)
    _install_manager(monkeypatch, manager)
    failure = AfterRunHookError([(DATASET_A.id, DATASET_A.name, RuntimeError("compaction"))])

    async def fake_propose(qa_rows, context_entries):
        return _curator_run(context_entries, [ProposedLesson(working_statement="Something.")])

    async def fake_accept(scope, proposed, context_entries):
        return [WrittenLesson(accept=True, statement="Keep reports concise.")], 0

    async def failing_step_publish(scope, accepted):
        raise failure

    monkeypatch.setattr(distill_module, "propose_lessons", fake_propose)
    monkeypatch.setattr(distill_module, "accept_proposed_lessons", fake_accept)
    monkeypatch.setattr(distill_module, "publish_distilled_lessons", failing_step_publish)

    with pytest.raises(AfterRunHookError) as raised:
        await distill_module.distill_session("s-1", dataset=DATASET_A.name, user=USER)

    assert raised.value is failure
    assert await get_distilled_entry_ids(manager, str(USER.id), "s-1", str(DATASET_A.id)) == {
        rows[0]["id"]
    }
