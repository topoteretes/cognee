"""improve() persists a session's pending agent traces as one document.

By default one run drains a session's whole backlog in one cognify run; the
step cap that splits it into windows is opt-in, and so is the cap on windows
per run. A stuck session is left out of the run's later passes so the others
keep draining.
"""

import importlib
import types
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.cache.models import SessionAgentTraceEntry
from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
from cognee.infrastructure.session.session_persist_watermark import TRACE_PERSIST_WATERMARK
from cognee.modules.improve import GraphCapabilities, ImproveConfig, ImproveRunInputs
from cognee.modules.improve.result import StageResult
from cognee.modules.improve.stage import execute_stage
from cognee.modules.improve.stages import PersistAgentTracesStage
from cognee.modules.pipelines.exceptions import CognifyFailedError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.tasks.memify.cognify_agent_trace_feedback import cognify_agent_trace_feedback
from cognee.tasks.memify.extract_agent_trace_feedbacks import extract_agent_trace_feedbacks


class FakeSessionManager:
    """In-memory trace store with the context rows the persist watermark lives in."""

    is_available = True

    def __init__(self):
        self.traces: dict[str, list[SessionAgentTraceEntry]] = {}
        self.context: dict[str, list[dict]] = {}

    def add_steps(self, session_id: str, count: int):
        steps = self.traces.setdefault(session_id, [])
        for _ in range(count):
            steps.append(
                SessionAgentTraceEntry(
                    trace_id=str(uuid4()),
                    origin_function="agent",
                    status="success",
                    session_feedback=f"{session_id} step {len(steps)}",
                )
            )

    async def get_agent_trace_count(self, *, user_id, session_id=None):
        return len(self.traces.get(session_id, []))

    async def get_agent_trace_session(self, *, user_id, session_id=None, last_n=None):
        return list(self.traces.get(session_id, []))

    async def get_session_context_entries(self, *, user_id, session_id=None, raise_on_error=False):
        return list(self.context.get(session_id, []))

    async def update_session_context_entry(self, *, user_id, entry_id, merge, session_id=None):
        for row in self.context.get(session_id, []):
            if row.get("id") == entry_id:
                row.update(merge)
                return True
        return False

    async def create_session_context_entry(self, *, user_id, entry_dump, session_id=None):
        self.context.setdefault(session_id, []).append(dict(entry_dump))
        return True


def _session_of(document: str) -> str:
    return document.split("\n", 1)[0].replace("Session ID: ", "")


class FakeCognee:
    """Records each document handed to add and each cognify run over them.

    A cognify run covers every document added since the previous completed
    run, as the real one does: a failed run is rolled back as a whole, its
    documents stay pending for the next run, and it raises CognifyFailedError
    with the cause (a budget rejection raises LLMPaymentRequiredError). It
    fails when its number is in ``fail_calls`` or when one of its documents
    belongs to a session in ``fail_sessions``. ``on_cognify`` runs on every
    build, to model an agent still writing steps.
    """

    def __init__(self, fail_calls=(), fail_sessions=(), on_cognify=None):
        self.documents: list[str] = []
        self.pending: list[str] = []
        self.runs: list[list[str]] = []  # sessions per cognify run
        self.fail_calls = set(fail_calls)
        self.fail_sessions = set(fail_sessions)
        self.on_cognify = on_cognify

    async def add(self, text, **kwargs):
        self.documents.append(text)
        self.pending.append(text)

    async def cognify(self, **kwargs):
        if self.on_cognify is not None:
            self.on_cognify()
        sessions = [_session_of(document) for document in self.pending]
        self.runs.append(sessions)
        if len(self.runs) in self.fail_calls or self.fail_sessions & set(sessions):
            # Rolled back: the documents stay pending, the next run picks them up.
            if self.error_class == "LLMPaymentRequiredError":
                raise LLMPaymentRequiredError(self.error_message)
            raise CognifyFailedError(
                dataset_name="docs",
                error_class=self.error_class,
                error_message=self.error_message,
            )
        self.pending = []
        return {}

    error_class = "RateLimitError"
    error_message = "rate limit exceeded"

    def steps_per_document(self) -> list[int]:
        return [len(document.split("\n\n", 1)[1].splitlines()) for document in self.documents]


@pytest.fixture
def store(monkeypatch):
    manager = FakeSessionManager()
    for module in (
        "cognee.infrastructure.session.get_session_manager",
        "cognee.tasks.memify.extract_agent_trace_feedbacks",
        "cognee.tasks.memify.cognify_agent_trace_feedback",
    ):
        monkeypatch.setattr(importlib.import_module(module), "get_session_manager", lambda: manager)
    return manager


def _use(monkeypatch, cognee_fake, after_pass=None, **settings) -> ImproveConfig:
    """Install the fake cognee and pipeline; returns the config to hand to the run.

    ``after_pass`` runs after each pipeline call, between the stage's passes.
    """
    config = ImproveConfig(**settings)
    for name in ("add", "cognify"):
        monkeypatch.setattr(
            importlib.import_module("cognee"), name, getattr(cognee_fake, name), raising=False
        )

    async def pipeline(
        *, user, session_ids, raw_trace_content, last_n_steps, persist_up_to=None, **kwargs
    ):
        """The memify pipeline's two real tasks, without the dataset and pipeline plumbing."""
        extract_module = importlib.import_module(
            "cognee.tasks.memify.extract_agent_trace_feedbacks"
        )
        monkeypatch.setattr(extract_module, "session_user", types.SimpleNamespace(get=lambda: user))
        windows = [
            window
            async for window in extract_agent_trace_feedbacks(
                [{}],
                session_ids=session_ids,
                raw_trace_content=raw_trace_content,
                last_n_steps=last_n_steps,
                persist_up_to=persist_up_to,
            )
        ]
        # The runner hands the enrichment task the pass's windows as one batch,
        # and re-raises what the task raises after recording the errored run.
        if windows:
            await cognify_agent_trace_feedback(windows, dataset_id=uuid4(), user=user)
        if after_pass is not None:
            await after_pass()
        return {}

    monkeypatch.setattr(
        importlib.import_module(
            "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph"
        ),
        "persist_agent_trace_feedbacks_in_knowledge_graph_pipeline",
        AsyncMock(side_effect=pipeline),
    )
    return config


def _inputs(config: ImproveConfig, session_ids=("s1",)):
    user = types.SimpleNamespace(id=uuid4())
    dataset = types.SimpleNamespace(id=uuid4(), name="docs", owner_id=user.id)
    return ImproveRunInputs(
        user=user,
        dataset_id=dataset.id,
        dataset=dataset,
        session_ids=tuple(session_ids),
        config=config,
        capabilities=GraphCapabilities.assume_supported(),
    )


async def _watermark(store, inputs, session_id="s1"):
    return await TRACE_PERSIST_WATERMARK.read_count(store, str(inputs.user.id), session_id)


@pytest.mark.asyncio
async def test_backlog_is_persisted_in_documents_of_at_most_the_window_size(store, monkeypatch):
    store.add_steps("s1", 95)
    cognee_fake = FakeCognee()
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert cognee_fake.steps_per_document() == [30, 30, 30, 5]
    assert await _watermark(store, inputs) == 95


@pytest.mark.asyncio
async def test_default_persists_a_sessions_whole_backlog_as_one_document(store, monkeypatch):
    """No step cap by default: one session, one document, one pass, whatever the backlog."""
    store.add_steps("s1", 95)
    cognee_fake = FakeCognee()
    inputs = _inputs(_use(monkeypatch, cognee_fake))

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert result.counts["windows"] == 1
    assert cognee_fake.steps_per_document() == [95]
    assert await _watermark(store, inputs) == 95


@pytest.mark.asyncio
async def test_default_run_drains_a_backlog_larger_than_the_old_cap(store, monkeypatch):
    """No cap by default: one run persists every pending step, however many windows."""
    store.add_steps("s1", 30 * 25)
    cognee_fake = FakeCognee()
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert result.counts["windows"] == 25
    assert await _watermark(store, inputs) == 750


@pytest.mark.asyncio
async def test_steps_written_during_the_run_wait_for_the_next_run(store, monkeypatch):
    """The run drains the backlog it started with. An agent that keeps writing
    cannot keep the run, and the dataset's improve lock, going."""
    store.add_steps("s1", 60)
    cognee_fake = FakeCognee(on_cognify=lambda: store.add_steps("s1", 30))
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert result.counts["windows"] == 2
    assert await _watermark(store, inputs) == 60
    assert await store.get_agent_trace_count(user_id=str(inputs.user.id), session_id="s1") == 120

    cognee_fake.on_cognify = None
    second = await PersistAgentTracesStage().run(inputs)

    assert second.status == "completed"
    assert await _watermark(store, inputs) == 120


@pytest.mark.asyncio
async def test_last_window_stops_at_the_starting_count_while_the_agent_writes(store, monkeypatch):
    """A backlog that is not a whole number of windows: the last window is cut at
    the count the run started with, not at the live count, so the run neither
    takes the agent's new steps nor mistakes the session for a stuck one."""
    store.add_steps("s1", 45)
    cognee_fake = FakeCognee(on_cognify=lambda: store.add_steps("s1", 30))
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert result.counts["windows"] == 2
    assert "sessions_stuck" not in result.counts
    assert cognee_fake.steps_per_document() == [30, 15]
    assert await _watermark(store, inputs) == 45

    cognee_fake.on_cognify = None
    second = await PersistAgentTracesStage().run(inputs)

    assert second.status == "completed"
    assert await _watermark(store, inputs) == 105


@pytest.mark.asyncio
async def test_watermark_moved_past_the_start_count_by_another_persister_is_done(
    store, monkeypatch
):
    """The agent's own periodic persist runs outside improve with no cut. When it
    moves the watermark past this run's starting count between two passes, the
    run is done with that session: not stale, no second drain, no watermark
    written backwards."""
    inputs_box = {}

    async def external_persist():
        user_id = str(inputs_box["inputs"].user.id)
        if await TRACE_PERSIST_WATERMARK.read_count(store, user_id, "s1") == 100:
            store.add_steps("s1", 30)
            await TRACE_PERSIST_WATERMARK.write_count(store, user_id, "s1", 130)

    store.add_steps("s1", 100)
    cognee_fake = FakeCognee()
    inputs = _inputs(
        _use(monkeypatch, cognee_fake, after_pass=external_persist, trace_window_steps=30)
    )
    inputs_box["inputs"] = inputs

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert result.counts["windows"] == 4
    assert cognee_fake.steps_per_document() == [30, 30, 30, 10]
    assert await _watermark(store, inputs) == 130


@pytest.mark.asyncio
async def test_failed_window_keeps_earlier_windows_and_stops_the_stage(store, monkeypatch):
    store.add_steps("s1", 95)
    cognee_fake = FakeCognee(fail_calls={2})
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    result = await execute_stage(PersistAgentTracesStage(), inputs)

    assert result.status == "errored"
    assert "rate limit exceeded" in (result.error or "")
    assert cognee_fake.steps_per_document() == [30, 30]
    assert await _watermark(store, inputs) == 30


@pytest.mark.asyncio
async def test_a_failed_pass_ends_the_run_with_its_cause(store, monkeypatch):
    """One cognify builds the pass. When it fails, the run is rolled back and the
    error raises through the memify run with its cause: nothing is built, no
    watermark moves, the stage reports the cause, and the next run retries."""
    store.add_steps("a", 95)
    store.add_steps("b", 95)
    cognee_fake = FakeCognee(fail_sessions={"a"})
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30), ("a", "b"))

    result = await execute_stage(PersistAgentTracesStage(), inputs)

    assert result.status == "errored"
    assert "rate limit exceeded" in (result.error or "")
    assert cognee_fake.runs == [["a", "b"]]
    assert await _watermark(store, inputs, "a") == 0
    assert await _watermark(store, inputs, "b") == 0


@pytest.mark.asyncio
async def test_a_pass_builds_its_windows_in_one_cognify_run(store, monkeypatch):
    """Ten sessions in a pass are one cognify run, not ten: the run builds them
    concurrently, so a pass costs about one window of LLM time."""
    session_ids = [f"s{i}" for i in range(10)]
    for session_id in session_ids:
        store.add_steps(session_id, 30)
    cognee_fake = FakeCognee()
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30), session_ids)

    result = await PersistAgentTracesStage().run(inputs)

    assert result.status == "completed"
    assert cognee_fake.runs == [session_ids]
    for session_id in session_ids:
        assert await _watermark(store, inputs, session_id) == 30


@pytest.mark.asyncio
async def test_exhausted_budget_stops_the_stage_and_is_classified(store, monkeypatch):
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
    from cognee.modules.improve.stage import execute_stage

    store.add_steps("s1", 95)
    cognee_fake = FakeCognee(fail_calls={2})
    cognee_fake.error_class = "LLMPaymentRequiredError"
    cognee_fake.error_message = "insufficient credits"
    inputs = _inputs(_use(monkeypatch, cognee_fake, trace_window_steps=30))

    with pytest.raises(LLMPaymentRequiredError):
        await PersistAgentTracesStage().run(inputs)

    # The first window was persisted; the second is retried once there is budget.
    assert await _watermark(store, inputs) == 30
    assert len(cognee_fake.documents) == 2

    # Through the orchestrator's stage runner the raise becomes an errored result
    # that classifies as budget exhaustion, so the later LLM stages are skipped.
    store.add_steps("s2", 10)
    broke = FakeCognee(fail_calls={1})
    broke.error_class = "LLMPaymentRequiredError"
    broke.error_message = "insufficient credits"
    result = await execute_stage(
        PersistAgentTracesStage(), _inputs(_use(monkeypatch, broke), ("s2",))
    )

    assert result.status == "errored"
    assert result.budget_exhausted


@pytest.mark.asyncio
async def test_a_pass_takes_a_bounded_number_of_sessions_in_turn(store, monkeypatch):
    """Each pass is one memify run holding the dataset lock, so it takes at most
    TRACE_SESSIONS_PER_PASS sessions; the least-served go first, so sessions late
    in the list are not left until the earlier ones have drained."""
    from cognee.modules.improve.stages import TRACE_SESSIONS_PER_PASS

    session_ids = [f"s{i}" for i in range(12)]
    for session_id in session_ids:
        store.add_steps(session_id, 60)
    inputs = _inputs(_use(monkeypatch, FakeCognee(), trace_window_steps=30), session_ids)
    pipeline = importlib.import_module(
        "cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph"
    ).persist_agent_trace_feedbacks_in_knowledge_graph_pipeline

    result = await PersistAgentTracesStage().run(inputs)

    passes = [call.kwargs["session_ids"] for call in pipeline.call_args_list]
    assert TRACE_SESSIONS_PER_PASS == 10
    assert [len(batch) for batch in passes] == [10, 10, 4]
    assert passes[1][:2] == ["s10", "s11"]
    assert result.status == "completed"
    assert result.counts["windows"] == 24
    for session_id in session_ids:
        assert await _watermark(store, inputs, session_id) == 60


@pytest.mark.asyncio
async def test_exhausted_budget_survives_the_real_pipeline_runner(store, monkeypatch):
    """Through the real run_tasks (storage faked), the task's 402 reaches the stage
    as a PipelineRunErrored that still names LLMPaymentRequiredError, so the
    errored stage classifies as budget exhaustion."""
    from cognee.infrastructure.llm.exceptions import LLMPaymentRequiredError
    from cognee.modules.pipelines.tasks.task import Task
    from cognee.tests.unit.modules.pipelines.test_run_tasks_rollback import (
        _FakeEngine,
        _no_op_context,
    )

    runner = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")
    store.add_steps("s1", 40)
    broke = FakeCognee(fail_calls={1})
    broke.error_class = "LLMPaymentRequiredError"
    broke.error_message = "insufficient credits"
    _use(monkeypatch, broke)
    dataset = types.SimpleNamespace(id=uuid4(), name="docs", owner_id=uuid4())
    user = types.SimpleNamespace(id=uuid4(), tenant_id=None)
    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.memify.extract_agent_trace_feedbacks"),
        "session_user",
        types.SimpleNamespace(get=lambda: user),
    )
    monkeypatch.setattr(runner, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(runner, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(
        runner,
        "log_pipeline_run_start",
        AsyncMock(return_value=types.SimpleNamespace(pipeline_run_id=uuid4())),
    )
    monkeypatch.setattr(runner, "log_pipeline_run_error", AsyncMock())
    monkeypatch.setattr(runner, "log_pipeline_run_progress", AsyncMock())
    monkeypatch.setattr(runner, "pipeline_run_telemetry_properties", lambda *a, **k: {})
    tasks = [
        Task(extract_agent_trace_feedbacks, session_ids=["s1"], last_n_steps=30),
        Task(cognify_agent_trace_feedback, dataset_id=dataset.id, user=user),
    ]

    runs = []
    with pytest.raises(LLMPaymentRequiredError):
        async for run_info in runner.run_tasks(tasks, dataset.id, [{}], user, "memify_pipeline"):
            runs.append(run_info)

    assert isinstance(runs[-1], PipelineRunErrored)
    assert runs[-1].error_class == "LLMPaymentRequiredError"
    stage_result = StageResult.from_pipeline_run("persist_agent_traces", {dataset.id: runs[-1]})
    assert stage_result.status == "errored"
    assert stage_result.budget_exhausted
    assert await TRACE_PERSIST_WATERMARK.read_count(store, str(user.id), "s1") == 0
    assert len(broke.documents) == 1


@pytest.mark.asyncio
async def test_window_cap_defers_the_rest_of_the_backlog_to_the_next_run(store, monkeypatch):
    store.add_steps("a", 95)
    store.add_steps("b", 95)
    cognee_fake = FakeCognee()
    inputs = _inputs(
        _use(monkeypatch, cognee_fake, trace_window_steps=30, trace_windows_per_run=3),
        ("a", "b"),
    )

    first = await PersistAgentTracesStage().run(inputs)

    assert first.status == "completed"
    assert first.counts["windows"] == 3
    assert len(cognee_fake.documents) == 3
    assert await _watermark(store, inputs, "a") + await _watermark(store, inputs, "b") == 90

    for _ in range(3):
        await PersistAgentTracesStage().run(inputs)

    assert await _watermark(store, inputs, "a") == 95
    assert await _watermark(store, inputs, "b") == 95
    assert max(cognee_fake.steps_per_document()) == 30


def test_window_steps_defaults_to_no_cap_and_rejects_negative():
    assert ImproveConfig(_env_file=None).trace_window_steps == 0
    with pytest.raises(ValueError):
        ImproveConfig(trace_window_steps=-1)


def test_windows_per_run_defaults_to_drain_and_rejects_negative():
    assert ImproveConfig(_env_file=None).trace_windows_per_run == 0
    with pytest.raises(ValueError):
        ImproveConfig(trace_windows_per_run=-1)
