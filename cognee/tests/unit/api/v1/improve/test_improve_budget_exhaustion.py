"""improve() stops at the first stage that fails on an exhausted LLM budget.

Every later stage would fail on it the same way, so the rest is recorded
``skipped: budget_exhausted`` and no further pass starts. The stage that hit it
stays ``errored``, and so does the run.

The first half drives the orchestrator with fake stages. The last test runs
the real session-bridge stage over the real memify pipeline runner, to prove
that a budget failure inside ``cognify_session`` reaches the orchestrator in a
shape it can classify.
"""

import importlib
import sys
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import cognee
from cognee.infrastructure.databases.cache.models import SessionQAEntry
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.session.session_persist_watermark import get_persisted_qa_count
from cognee.modules.improve import (
    REASON_ABORTED_BY_FATAL_STAGE,
    REASON_BUDGET_EXHAUSTED,
    ImproveResult,
)
from cognee.modules.improve.result import StageResult
from cognee.modules.improve.stages import PersistSessionQAStage
from cognee.modules.observability import COGNEE_IMPROVE_STAGES
from cognee.modules.pipelines.models import OperationOutcome
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)

from .conftest import FakeStage

session_lock = importlib.import_module("cognee.infrastructure.locks.session_lock")

BUDGET_SENTENCE = "Budget has been exceeded! Current cost: 20.0, Max budget: 20.0"


@pytest.fixture(autouse=True)
def _clean_registry():
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()
    yield
    session_lock._improving_sessions.clear()
    session_lock._rerun_requested.clear()


def _budget_error() -> LLMPaymentRequiredError:
    return LLMPaymentRequiredError(f"LLM budget exhausted: {BUDGET_SENTENCE}")


def _budget_errored_run(harness) -> PipelineRunErrored:
    """What a pipeline hands back when it reported the failure instead of raising."""
    return PipelineRunErrored(
        pipeline_run_id=uuid.uuid4(),
        dataset_id=harness.dataset.id,
        dataset_name="docs",
        error_class="LLMPaymentRequiredError",
        error_message="LLMPaymentRequiredError: LLM provider requires payment or token budget "
        "is exhausted. (Status code: 402)",
    )


def _outcomes(stages: list[StageResult]) -> list[tuple[str, str, str | None]]:
    return [(stage.stage, stage.status, stage.reason) for stage in stages]


async def _assert_claim_is_free(keys) -> None:
    assert await session_lock.try_acquire_improve_lock_many(keys)
    await session_lock.release_improve_lock_many(keys)


@pytest.mark.asyncio
async def test_stage_that_raises_on_the_budget_stops_the_run_and_skips_the_rest(harness):
    calls = []
    harness.use_stages(
        [
            FakeStage("first", calls=calls),
            FakeStage("broke", run=lambda _i: _budget_error(), calls=calls),
            FakeStage("after_1", calls=calls),
            FakeStage("after_2", calls=calls),
        ]
    )

    result = await harness.improve()

    assert isinstance(result, ImproveResult)
    assert calls == ["first", "broke"]
    assert _outcomes(result.stages) == [
        ("first", "completed", None),
        ("broke", "errored", None),
        ("after_1", "skipped", REASON_BUDGET_EXHAUSTED),
        ("after_2", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    # The stage that hit the budget is recorded as it ended, error and all.
    assert BUDGET_SENTENCE in result.stages[1].error
    # The run is honest about it: errored, and the row says failed. ``error``
    # stays the fatal-stage field; nothing fatal happened here.
    assert result.status == "errored"
    assert result.error is None
    assert result.finished is True
    assert harness.operations[-1].outcome == OperationOutcome.FAILED
    assert harness.span.attributes[COGNEE_IMPROVE_STAGES] == (
        "first=completed,broke=errored,after_1=skipped,after_2=skipped"
    )
    await _assert_claim_is_free([f"dataset:{harness.dataset.id}"])


@pytest.mark.asyncio
async def test_stage_whose_pipeline_reported_the_budget_without_raising_also_stops_the_run(
    harness,
):
    """No exception to classify: only the errored run's class and message."""
    calls = []
    errored_run = _budget_errored_run(harness)
    harness.use_stages(
        [
            FakeStage(
                "broke",
                run=lambda _i: StageResult.from_pipeline_run(
                    "broke", {harness.dataset.id: errored_run}
                ),
                calls=calls,
            ),
            FakeStage("after", calls=calls),
        ]
    )

    result = await harness.improve()

    assert calls == ["broke"]
    assert result.stages[0].exception is None
    assert _outcomes(result.stages) == [
        ("broke", "errored", None),
        ("after", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert result.status == "errored"


@pytest.mark.asyncio
async def test_fatal_stage_on_the_budget_still_raises_and_names_the_budget(harness):
    """persist_session_qa keeps its contract — the run stops and raises, with
    the partial result on the exception — but the stages it cut off say why."""
    calls = []
    boom = _budget_error()
    harness.use_stages(
        [
            FakeStage("first", calls=calls),
            FakeStage("fatal_one", fatal=True, run=lambda _i: boom, calls=calls),
            FakeStage("after_fatal", calls=calls),
            FakeStage("last", calls=calls),
        ]
    )

    with pytest.raises(LLMPaymentRequiredError) as excinfo:
        await harness.improve()

    assert excinfo.value is boom
    assert calls == ["first", "fatal_one"]
    partial = excinfo.value.improve_result
    assert _outcomes(partial.stages) == [
        ("first", "completed", None),
        ("fatal_one", "errored", None),
        ("after_fatal", "skipped", REASON_BUDGET_EXHAUSTED),
        ("last", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert partial.error == partial.stages[1].error
    assert partial.status == "errored"
    await _assert_claim_is_free([f"dataset:{harness.dataset.id}"])


@pytest.mark.asyncio
async def test_fatal_stage_whose_pipeline_reported_the_budget_raises_the_typed_error(harness):
    """Nothing was raised for the orchestrator to re-raise: the stage's pipeline
    only reported the failure. The abort must still carry the 402, or the HTTP
    layer and RememberResult.improve_error are left string-matching the text."""
    errored_run = _budget_errored_run(harness)
    harness.use_stages(
        [
            FakeStage(
                "fatal_one",
                fatal=True,
                run=lambda _i: StageResult.from_pipeline_run(
                    "fatal_one", {harness.dataset.id: errored_run}
                ),
            ),
            FakeStage("after_fatal"),
        ]
    )

    with pytest.raises(LLMPaymentRequiredError) as excinfo:
        await harness.improve()

    assert excinfo.value.status_code == 402
    # What the stage reported is kept in the message.
    assert "fatal_one" in excinfo.value.message
    assert "LLM provider requires payment" in excinfo.value.message
    partial = excinfo.value.improve_result
    assert partial.stages[0].exception is None
    assert _outcomes(partial.stages) == [
        ("fatal_one", "errored", None),
        ("after_fatal", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert partial.error == partial.stages[0].error


@pytest.mark.asyncio
async def test_fatal_stage_whose_pipeline_reported_another_failure_stays_generic(harness):
    """Only a budget failure is typed as one; any other reported failure keeps
    the generic abort error and its 500."""
    errored_run = PipelineRunErrored(
        pipeline_run_id=uuid.uuid4(),
        dataset_id=harness.dataset.id,
        dataset_name="docs",
        error_class="RuntimeError",
        error_message="RuntimeError: graph store unreachable",
    )
    harness.use_stages(
        [
            FakeStage(
                "fatal_one",
                fatal=True,
                run=lambda _i: StageResult.from_pipeline_run(
                    "fatal_one", {harness.dataset.id: errored_run}
                ),
            ),
            FakeStage("after_fatal"),
        ]
    )

    with pytest.raises(Exception) as excinfo:
        await harness.improve()

    assert not isinstance(excinfo.value, LLMPaymentRequiredError)
    assert excinfo.value.name == "ImproveFatalStageError"
    assert excinfo.value.status_code != 402
    assert _outcomes(excinfo.value.improve_result.stages)[1] == (
        "after_fatal",
        "skipped",
        REASON_ABORTED_BY_FATAL_STAGE,
    )


@pytest.mark.asyncio
async def test_fatal_stage_failing_for_another_reason_keeps_the_abort_reason(harness):
    """The budget reason is only for budget failures, in both shapes."""
    harness.use_stages(
        [
            FakeStage("fatal_one", fatal=True, run=lambda _i: RuntimeError("persist failed")),
            FakeStage("after_fatal"),
        ]
    )

    with pytest.raises(RuntimeError) as excinfo:
        await harness.improve()

    partial = excinfo.value.improve_result
    assert partial.stages[0].budget_exhausted is False
    assert _outcomes(partial.stages)[1] == ("after_fatal", "skipped", REASON_ABORTED_BY_FATAL_STAGE)


@pytest.mark.asyncio
async def test_budget_stop_starts_no_rerun_pass_and_releases_the_claim_once(harness, monkeypatch):
    """A loser asked this run for one more pass. That pass would fail on the
    same budget, so it never starts: the claim is released by the plain
    release — exactly once — and the request is left to the next claimant."""
    improve_mod = harness.improve_mod
    calls = []
    releases = []
    session_key = f"session:{harness.user.id}:chat_1"
    real_release_or_rerun = improve_mod.release_or_rerun_improve_lock_many
    real_release = improve_mod.release_improve_lock_many

    async def spy_release_or_rerun(keys, *, rerun_keys):
        releases.append("release_or_rerun")
        return await real_release_or_rerun(keys, rerun_keys=rerun_keys)

    async def spy_release(keys):
        releases.append("release")
        await real_release(keys)

    monkeypatch.setattr(improve_mod, "release_or_rerun_improve_lock_many", spy_release_or_rerun)
    monkeypatch.setattr(improve_mod, "release_improve_lock_many", spy_release)

    async def asked_for_one_more(_inputs):
        assert await session_lock.request_improve_rerun_many([session_key])
        return StageResult.completed("asked", items=1)

    harness.use_stages(
        [
            FakeStage("asked", run=asked_for_one_more, calls=calls),
            FakeStage("broke", run=lambda _i: _budget_error(), calls=calls),
            FakeStage("after", calls=calls),
        ]
    )

    result = await harness.improve(session_ids=["chat_1"])

    assert calls == ["asked", "broke"]  # one pass, cut short
    assert result.rerun_passes == []
    assert _outcomes(result.stages) == [
        ("asked", "completed", None),
        ("broke", "errored", None),
        ("after", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert result.stage_summary() == "asked=completed,broke=errored,after=skipped"
    assert releases == ["release"]
    # The request survives the release and is cleared by the next claim, whose
    # full pass covers it.
    assert session_key in session_lock._rerun_requested
    keys = session_lock.improve_lock_keys(["chat_1"], harness.dataset.id, harness.user.id)
    assert await session_lock.try_acquire_improve_lock_many(keys)
    assert session_key not in session_lock._rerun_requested


@pytest.mark.asyncio
async def test_budget_failure_in_the_last_stage_starts_no_rerun_pass(harness):
    """Nothing is left to skip, and still no further pass."""
    calls = []
    session_key = f"session:{harness.user.id}:chat_1"

    async def broke_after_a_rerun_request(_inputs):
        await session_lock.request_improve_rerun_many([session_key])
        raise _budget_error()

    harness.use_stages(
        [
            FakeStage("first", calls=calls),
            FakeStage("broke", run=broke_after_a_rerun_request, calls=calls),
        ]
    )

    result = await harness.improve(session_ids=["chat_1"])

    assert calls == ["first", "broke"]
    assert result.rerun_passes == []
    assert _outcomes(result.stages) == [("first", "completed", None), ("broke", "errored", None)]
    assert result.status == "errored"


@pytest.mark.asyncio
async def test_budget_failure_inside_a_rerun_pass_stops_that_pass(harness):
    """The stop applies to whichever pass hits the budget; its skipped stages
    are recorded in that pass, and no third pass follows."""
    calls = []
    session_key = f"session:{harness.user.id}:chat_1"

    async def fine_then_broke(_inputs):
        # Every call asks for one more pass, as a stream of losers would.
        await session_lock.request_improve_rerun_many([session_key])
        if calls.count("flaky") == 1:
            return StageResult.completed("flaky", items=1)
        raise _budget_error()

    harness.use_stages(
        [FakeStage("flaky", run=fine_then_broke, calls=calls), FakeStage("after", calls=calls)]
    )

    result = await harness.improve(session_ids=["chat_1"])

    assert calls == ["flaky", "after", "flaky"]
    assert _outcomes(result.stages) == [("flaky", "completed", None), ("after", "completed", None)]
    assert len(result.rerun_passes) == 1
    assert _outcomes(result.rerun_passes[0]) == [
        ("flaky", "errored", None),
        ("after", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert result.status == "errored"


@pytest.mark.asyncio
async def test_background_run_stops_on_the_budget_without_an_error_to_report(harness):
    """Nothing raised, so the detached run closes its row with no exception;
    the failed outcome comes from the errored stage."""
    harness.use_stages(
        [
            FakeStage("broke", run=lambda _i: _budget_error()),
            FakeStage("after"),
        ]
    )

    result = await harness.improve(run_in_background=True)
    await result.wait()

    assert _outcomes(result.stages) == [
        ("broke", "errored", None),
        ("after", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert result.status == "errored"
    assert result.error is None
    assert [call["error"] for call in harness.finish_calls] == [None]
    assert harness.operations[-1].outcome == OperationOutcome.FAILED
    await _assert_claim_is_free([f"dataset:{harness.dataset.id}"])


# --- the chain: cognify_session -> memify pipeline -> stage -> orchestrator ---


class _FakeSessionManager:
    """The SessionManager surface the persist stage and its two tasks use."""

    is_available = True

    def __init__(self):
        self.qa: dict[tuple[str, str], list[SessionQAEntry]] = {}
        self.context: dict[tuple[str, str], list[dict]] = {}

    def add_entry(self, user_id: str, session_id: str, question: str, answer: str):
        self.qa.setdefault((user_id, session_id), []).append(
            SessionQAEntry(
                time="2026-09-30T00:00:00+00:00",
                question=question,
                context="",
                answer=answer,
                qa_id=str(uuid.uuid4()),
            )
        )

    async def get_session(self, *, user_id, session_id=None, formatted=False, **_):
        return list(self.qa.get((user_id, session_id), []))

    async def get_session_context_entries(self, *, user_id, session_id=None, raise_on_error=False):
        return list(self.context.get((user_id, session_id), []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.context.get((user_id, session_id), []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.context.setdefault((user_id, session_id), []).append(dict(entry_dump))
        return True


class _DatasetSession:
    def __init__(self, dataset):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@asynccontextmanager
async def _no_database_context(*_args, **_kwargs):
    yield


@pytest.fixture
def session_bridge(harness, monkeypatch):
    """Everything below improve() is real except databases, auth and the LLM.

    Real: the persist stage, ``persist_sessions_in_knowledge_graph_pipeline``,
    ``memify``, ``run_pipeline`` and ``run_tasks``, the task runner, and both
    tasks (``extract_user_sessions``, ``cognify_session``) with their
    watermarks. Stubbed: dataset authorization, setup and migrations, the
    relational session and run-log writers ``run_tasks`` uses, the session
    cache (in memory), and ``cognee.add`` / ``cognee.cognify``.
    """
    # The packages re-export functions under their module names, so the module
    # objects have to come from sys.modules, not from a dotted attribute walk.
    importlib.import_module("cognee.memify_pipelines.persist_sessions_in_knowledge_graph")
    importlib.import_module("cognee.modules.memify.memify")
    importlib.import_module("cognee.modules.pipelines.operations.pipeline")
    persist_module = sys.modules["cognee.memify_pipelines.persist_sessions_in_knowledge_graph"]
    memify_module = sys.modules["cognee.modules.memify.memify"]
    pipeline_module = sys.modules["cognee.modules.pipelines.operations.pipeline"]
    run_tasks_module = sys.modules["cognee.modules.pipelines.operations.run_tasks"]
    migrations_module = importlib.import_module("cognee.modules.migrations.startup")
    session_manager_module = sys.modules["cognee.infrastructure.session.get_session_manager"]
    extract_module = sys.modules["cognee.tasks.memify.extract_user_sessions"]
    cognify_session_module = sys.modules["cognee.tasks.memify.cognify_session"]

    # The real task runner sends telemetry for every task it runs; keep the
    # test from doing that whatever environment it is started in.
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")

    dataset, user = harness.dataset, harness.user

    async def authorized_datasets(*_args, **_kwargs):
        return [dataset]

    async def resolved_datasets(_datasets, _user):
        return user, [dataset]

    monkeypatch.setattr(persist_module, "get_authorized_existing_datasets", authorized_datasets)
    monkeypatch.setattr(memify_module, "setup", AsyncMock(return_value=None))
    monkeypatch.setattr(memify_module, "resolve_authorized_user_datasets", resolved_datasets)
    monkeypatch.setattr(migrations_module, "run_migrations_and_block", AsyncMock())
    monkeypatch.setattr(pipeline_module, "setup_and_check_environment", AsyncMock())
    monkeypatch.setattr(pipeline_module, "resolve_authorized_user_datasets", resolved_datasets)

    run_log = SimpleNamespace(errors=[], completed=[])

    async def log_start(*_args, **_kwargs):
        return SimpleNamespace(pipeline_run_id=uuid.uuid4())

    async def log_error(_run_id, _pipeline_id, pipeline_name, _dataset_id, _data, error, **_kw):
        run_log.errors.append((pipeline_name, error))

    async def log_complete(_run_id, _pipeline_id, pipeline_name, *_args, **_kwargs):
        run_log.completed.append(pipeline_name)

    monkeypatch.setattr(
        run_tasks_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _DatasetSession(dataset)),
    )
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", log_error)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", log_complete)
    monkeypatch.setattr(
        run_tasks_module, "set_database_global_context_variables", _no_database_context
    )
    monkeypatch.setattr(
        run_tasks_module, "get_graph_engine", AsyncMock(return_value=SimpleNamespace())
    )

    sessions = _FakeSessionManager()
    for module in (session_manager_module, extract_module, cognify_session_module):
        monkeypatch.setattr(module, "get_session_manager", lambda: sessions)

    added = []

    async def fake_add(text, *_args, **_kwargs):
        added.append(text)

    monkeypatch.setattr(cognee, "add", fake_add)

    return SimpleNamespace(sessions=sessions, run_log=run_log, added=added)


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_cognify", ["returns_errored_run", "raises"])
async def test_budget_failure_in_cognify_session_stops_the_improve_run(
    harness, session_bridge, monkeypatch, nested_cognify
):
    """Two sessions to bridge; the second one's cognify runs out of budget.

    ``cognify(raise_on_error=False)`` has two ways to say so, depending on
    ``RAISE_INCREMENTAL_LOADING_ERRORS``: it hands back an errored run, or it
    raises the adapter's error. Either way the failure leaves
    ``cognify_session`` as LLMPaymentRequiredError, the memify runner re-raises
    a task's exception unchanged, the stage keeps it, and the orchestrator
    skips every remaining stage with ``budget_exhausted``.
    """
    user_id = str(harness.user.id)
    session_bridge.sessions.add_entry(user_id, "chat_1", "first question", "first answer")
    session_bridge.sessions.add_entry(user_id, "chat_2", "second question", "second answer")

    run_ids = {"dataset_id": harness.dataset.id, "dataset_name": "docs"}
    cognify_calls = []

    async def fake_cognify(*_args, **kwargs):
        cognify_calls.append(kwargs)
        if len(cognify_calls) == 1:
            return {
                harness.dataset.id: PipelineRunCompleted(pipeline_run_id=uuid.uuid4(), **run_ids)
            }
        if nested_cognify == "raises":
            raise _budget_error()
        return {
            harness.dataset.id: PipelineRunErrored(
                pipeline_run_id=uuid.uuid4(),
                error_class="LLMPaymentRequiredError",
                error_message=f"LLMPaymentRequiredError: LLM budget exhausted: {BUDGET_SENTENCE} "
                "(Status code: 402)",
                **run_ids,
            )
        }

    monkeypatch.setattr(cognee, "cognify", fake_cognify)

    later_stage_calls = []
    harness.use_stages(
        [
            FakeStage("feedback_weights", calls=later_stage_calls),
            PersistSessionQAStage(),
            FakeStage("persist_agent_traces", calls=later_stage_calls),
            FakeStage("distill_sessions", calls=later_stage_calls),
            FakeStage("triplet_enrichment", calls=later_stage_calls),
        ]
    )

    # The session-bridge stage is the fatal one, so the run raises — with the
    # budget error itself, not a wrapper around it.
    with pytest.raises(LLMPaymentRequiredError) as excinfo:
        await harness.improve(session_ids=["chat_1", "chat_2"])

    assert excinfo.value.status_code == 402
    assert BUDGET_SENTENCE in excinfo.value.message
    assert "Failed to cognify session data" not in str(excinfo.value)

    # What the runner did with the task's exception: recorded the memify run as
    # errored under the budget error's own class, and re-raised it.
    assert [(name, type(error).__name__) for name, error in session_bridge.run_log.errors] == [
        ("memify_pipeline", "LLMPaymentRequiredError")
    ]
    assert session_bridge.run_log.completed == []

    partial = excinfo.value.improve_result
    assert _outcomes(partial.stages) == [
        ("feedback_weights", "completed", None),
        ("persist_session_qa", "errored", None),
        ("persist_agent_traces", "skipped", REASON_BUDGET_EXHAUSTED),
        ("distill_sessions", "skipped", REASON_BUDGET_EXHAUSTED),
        ("triplet_enrichment", "skipped", REASON_BUDGET_EXHAUSTED),
    ]
    assert partial.stages[1].exception is excinfo.value
    assert partial.status == "errored"
    assert later_stage_calls == ["feedback_weights"]

    # Both windows were added and built once each; nothing was retried.
    assert len(session_bridge.added) == 2
    assert len(cognify_calls) == 2
    assert all(call["raise_on_error"] is False for call in cognify_calls)
    # The session persisted before the failure keeps its advanced watermark;
    # the one that failed stays put, so the next improve picks it up.
    assert await get_persisted_qa_count(session_bridge.sessions, user_id, "chat_1") == 1
    assert await get_persisted_qa_count(session_bridge.sessions, user_id, "chat_2") == 0
    keys = session_lock.improve_lock_keys(["chat_1", "chat_2"], harness.dataset.id, harness.user.id)
    await _assert_claim_is_free(keys)
