"""distill_session orchestration with the LLM gates mocked at the gateway level.

Covers what the helper-level tests miss: accepted lessons get published, rejected
lessons don't, sessions with no gated entries short-circuit before any LLM call,
and one failing curator batch drops only its own proposals.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cognee.modules.session_distillation import distill as distill_module
from cognee.modules.session_distillation.distill import distill_session, propose_lessons
from cognee.modules.session_distillation.models import (
    CuratorBatchOutput,
    ProposedLesson,
    WrittenLesson,
)


def _scope():
    return distill_module.SessionDistillationScope(
        session_id="s1",
        user=SimpleNamespace(id=uuid4()),
        dataset=SimpleNamespace(id=uuid4(), owner_id=uuid4()),
    )


def _entry(entry_id="e1", confidence=0.9):
    return SimpleNamespace(
        id=entry_id,
        harmful_count=0,
        helpful_count=1,
        confidence=confidence,
        content="candidate content",
        context_profile="qa",
        section="lessons_learned",
        created_at="2026-01-01T00:00:00+00:00",
    )


def _gateway(curator_lessons, written_lesson):
    """Route structured-output calls by response model: curator vs writer."""

    async def dispatch(*, text_input, system_prompt, response_model):
        if response_model is CuratorBatchOutput:
            return CuratorBatchOutput(lessons=curator_lessons)
        return written_lesson

    return AsyncMock(side_effect=dispatch)


def _wire(monkeypatch, scope, entries, curator_lessons, written_lesson):
    monkeypatch.setattr(distill_module, "resolve_distillation_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(
        distill_module,
        "load_distillable_session_inputs",
        AsyncMock(return_value=([{"question": "q", "answer": "a", "time": "t"}], entries)),
    )
    # Watermark store: first run has no state.
    monkeypatch.setattr(distill_module, "_load_processed_entry_ids", AsyncMock(return_value=set()))
    monkeypatch.setattr(distill_module, "_save_processed_entry_ids", AsyncMock())
    monkeypatch.setattr(
        distill_module.LLMGateway,
        "acreate_structured_output",
        _gateway(curator_lessons, written_lesson),
    )
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda _name: "prompt")

    vector_engine = MagicMock()
    vector_engine.search = AsyncMock(return_value=[])
    monkeypatch.setattr(
        distill_module, "get_vector_engine_async", AsyncMock(return_value=vector_engine)
    )

    class _NoopContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(
        distill_module, "set_database_global_context_variables", lambda *a, **k: _NoopContext()
    )

    import cognee.api.v1.add as add_module
    import cognee.api.v1.cognify as cognify_module

    add_spy = AsyncMock()
    cognify_spy = AsyncMock()
    monkeypatch.setattr(add_module, "add", add_spy)
    monkeypatch.setattr(cognify_module, "cognify", cognify_spy)
    return add_spy, cognify_spy


@pytest.mark.asyncio
async def test_accepted_lesson_is_published_with_node_sets(monkeypatch):
    scope = _scope()
    add_spy, cognify_spy = _wire(
        monkeypatch,
        scope,
        entries=[_entry()],
        curator_lessons=[ProposedLesson(working_statement="Use uv.", member_entry_ids=["e1"])],
        written_lesson=WrittenLesson(
            accept=True, statement="Always use uv for installs.", entities=["uv"]
        ),
    )

    result = await distill_session("s1", dataset="docs")

    assert result.status == "completed"
    assert len(result.documents) == 1
    assert "Always use uv for installs." in result.documents[0]
    assert "Entities: uv" in result.documents[0]
    add_spy.assert_awaited_once()
    node_set = add_spy.await_args.kwargs["node_set"]
    assert "session_learnings" in node_set
    assert "session_learnings:s1" in node_set
    cognify_spy.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejected_lesson_publishes_nothing(monkeypatch):
    scope = _scope()
    add_spy, cognify_spy = _wire(
        monkeypatch,
        scope,
        entries=[_entry()],
        curator_lessons=[ProposedLesson(working_statement="Old news.")],
        written_lesson=WrittenLesson(accept=False, reason="already_known"),
    )

    result = await distill_session("s1", dataset="docs")

    assert result.status == "no_accepted_lessons"
    assert result.documents == []
    add_spy.assert_not_awaited()
    cognify_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_gated_entries_short_circuits_before_llm(monkeypatch):
    scope = _scope()
    llm_spy = AsyncMock(side_effect=AssertionError("LLM must not run"))
    monkeypatch.setattr(distill_module, "resolve_distillation_scope", AsyncMock(return_value=scope))
    monkeypatch.setattr(
        distill_module,
        "load_distillable_session_inputs",
        AsyncMock(return_value=([{"question": "q", "answer": "a"}], [])),
    )
    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", llm_spy)

    result = await distill_session("s1", dataset="docs")

    assert result.status == "no_gated_entries"
    llm_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_failing_curator_batch_drops_only_its_own_work(monkeypatch):
    """Fail-open per batch: the second batch's proposals survive the first's crash."""
    calls = {"count": 0}

    async def flaky(*, text_input, system_prompt, response_model):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("llm down")
        return CuratorBatchOutput(
            lessons=[ProposedLesson(working_statement=f"Lesson {calls['count']}.")]
        )

    monkeypatch.setattr(distill_module.LLMGateway, "acreate_structured_output", flaky)
    monkeypatch.setattr(distill_module, "read_query_prompt", lambda _name: "prompt")
    # Two batches: more blocks than CURATOR_BLOCKS_PER_BATCH forces a second batch.
    qa_rows = [
        {"question": f"q{i}", "answer": f"a{i}", "time": f"2026-01-01T00:0{i}:00+00:00"}
        for i in range(distill_module.CURATOR_BLOCKS_PER_BATCH + 1)
    ]

    proposed = await propose_lessons(qa_rows, [])

    assert [lesson.working_statement for lesson in proposed] == ["Lesson 2."]
