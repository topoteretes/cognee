"""Orchestration tests for ``run_memory_score``: who it reads, and what it spends.

Both properties here are invisible from the outside once a run has finished, so
they are asserted at the seams the run calls out through:

* real questions are read for ONE user, never tenant-wide — query text is a
  member's search history and the run document hands it back verbatim;
* the spend parameters are clamped inside the run too, so a scheduler or SDK
  caller that never passes through FastAPI validation is bounded as well.
"""

import asyncio
from contextlib import asynccontextmanager
from importlib import import_module
from types import ModuleType, SimpleNamespace
from uuid import uuid4

from cognee.modules.memory_score.methods.build_topics import TopicPlan
from cognee.modules.memory_score.models import MemoryScoreRunStatus

run_module = import_module("cognee.modules.memory_score.methods.run_memory_score")

TENANT_ID = uuid4()
DATASET_ID = uuid4()
DATASET_OWNER_ID = uuid4()
CALLER_ID = uuid4()


class _Recorder:
    """Captures what the run asked the outside world for."""

    def __init__(self):
        self.query_reads = []
        self.generation_targets = []
        self.updates = []
        self.contexts = []


def _install(monkeypatch, recorder, *, topic_plan, run_id):
    """Stub every boundary ``run_memory_score`` crosses, recording the interesting ones."""

    async def resolve(_tenant_id, dataset_id, _requesting_user_id=None):
        return SimpleNamespace(id=dataset_id, owner_id=DATASET_OWNER_ID)

    async def claim(_tenant_id):
        return run_id

    async def update_run(_run_id, **fields):
        recorder.updates.append(fields)

    @asynccontextmanager
    async def dataset_context(dataset_id, user_id):
        recorder.contexts.append((dataset_id, user_id))
        yield

    async def detect_schema():
        return False

    async def get_queries(user_id, limit):
        recorder.query_reads.append((user_id, limit))
        return []

    async def build_topics(_real_question_texts):
        return topic_plan

    async def generate_questions(_plan, target_count):
        recorder.generation_targets.append(target_count)
        return []

    async def persist(_run_id, _rows):
        return None

    monkeypatch.setattr(run_module, "resolve_memory_score_dataset", resolve)
    monkeypatch.setattr(run_module, "_claim_initiated_run", claim)
    monkeypatch.setattr(run_module, "_update_run", update_run)
    monkeypatch.setattr(run_module, "set_database_global_context_variables", dataset_context)
    monkeypatch.setattr(run_module, "_detect_schema_defined", detect_schema)
    monkeypatch.setattr(run_module, "get_queries", get_queries)
    monkeypatch.setattr(run_module, "build_topics", build_topics)
    monkeypatch.setattr(run_module, "generate_questions", generate_questions)
    monkeypatch.setattr(run_module, "_persist_questions", persist)
    monkeypatch.setattr(run_module, "AnswerGeneratorExecutor", lambda *a, **k: object())
    monkeypatch.setattr(run_module, "DirectLLMEvalAdapter", lambda *a, **k: object())
    monkeypatch.setattr(run_module, "GroundednessAdapter", lambda *a, **k: object())


def test_patched_boundaries_are_callable_before_any_monkeypatching():
    """Guard against a stub masking a broken import.

    Every name `_install` replaces is asserted to be a real callable as imported,
    because monkeypatching hides the failure otherwise. This caught
    `from cognee.modules.search.operations import get_queries` binding the
    submodule rather than the function — the package does not re-export it, so
    every call raised "'module' object is not callable" while the stubbed tests
    passed. A live run found it; this test is what should have.
    """
    for name in (
        "get_queries",
        "build_topics",
        "generate_questions",
        "resolve_memory_score_dataset",
        "set_database_global_context_variables",
        "AnswerGeneratorExecutor",
        "DirectLLMEvalAdapter",
        "GroundednessAdapter",
    ):
        attribute = getattr(run_module, name)
        assert not isinstance(attribute, ModuleType), (
            f"run_memory_score.{name} is a module, not a callable — the import binds "
            f"the submodule instead of the object inside it"
        )
        assert callable(attribute), f"run_memory_score.{name} is not callable"


def _scoreable_plan():
    return TopicPlan(topics=[], chunk_count=100, below_data_floor=False, floor_reason=None)


def _gated_plan():
    return TopicPlan(
        topics=[],
        chunk_count=4,
        below_data_floor=True,
        floor_reason="4 document chunks, 50 required",
    )


def test_real_questions_are_read_for_the_triggering_user_only(monkeypatch):
    """Tenant-wide would show every member their colleagues' search history."""
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_scoreable_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=CALLER_ID,
            synthetic_target=10,
            real_question_limit=5,
        )
    )

    assert recorder.query_reads == [(CALLER_ID, 5)]


def test_a_scheduled_run_falls_back_to_the_dataset_owner(monkeypatch):
    """No acting user, but still someone the report is for."""
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_scoreable_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=None,
            synthetic_target=10,
            real_question_limit=5,
        )
    )

    assert recorder.query_reads == [(DATASET_OWNER_ID, 5)]


def test_spend_parameters_are_clamped_inside_the_run(monkeypatch):
    """A caller bypassing HTTP validation must not commit unbounded LLM spend."""
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_scoreable_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=CALLER_ID,
            synthetic_target=1_000_000,
            real_question_limit=10_000,
        )
    )

    assert recorder.generation_targets == [run_module.MAX_SYNTHETIC_TARGET]
    assert recorder.query_reads == [(CALLER_ID, run_module.MAX_REAL_QUESTION_LIMIT)]


def test_negative_spend_parameters_are_floored_at_zero(monkeypatch):
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_scoreable_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=CALLER_ID,
            synthetic_target=-1,
            real_question_limit=-1,
        )
    )

    assert recorder.generation_targets == [0]
    assert recorder.query_reads == [], "a zero limit reads no query history at all"


def test_the_run_enters_the_dataset_context_as_its_owner(monkeypatch):
    """Not as the caller: a scheduled run has no user, and the graph is the owner's."""
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_scoreable_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=CALLER_ID,
        )
    )

    assert recorder.contexts == [(DATASET_ID, DATASET_OWNER_ID)]


def test_a_gated_run_spends_nothing_on_generation(monkeypatch):
    """Below the data floor: persisted as skipped, no question ever generated."""
    recorder = _Recorder()
    _install(monkeypatch, recorder, topic_plan=_gated_plan(), run_id=uuid4())

    asyncio.run(
        run_module.run_memory_score(
            tenant_id=TENANT_ID,
            dataset_id=DATASET_ID,
            triggered_by_user_id=CALLER_ID,
            synthetic_target=100,
        )
    )

    assert recorder.generation_targets == []
    final_update = recorder.updates[-1]
    assert final_update["status"] == MemoryScoreRunStatus.SKIPPED_INSUFFICIENT_DATA
    assert final_update["below_data_floor"] is True
    assert final_update["floor_reason"] == "4 document chunks, 50 required"
