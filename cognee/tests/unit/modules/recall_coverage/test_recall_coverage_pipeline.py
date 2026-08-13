"""Guards on the recall-coverage pipeline — spec section 2, all four phases.

What this file pins down:

* **A valid label with no traffic completes as an EMPTY run**, with
  ``memory_score: null`` and no question rows — not an error, not a failure. It is
  the *expected* outcome for an agent that has asked nothing in the window, so it
  has to read as "nothing asked yet".
* **The in-flight guard is per ``(owner, agent_label)``**, and the run row is
  written before the coroutine is scheduled — otherwise the guard cannot see it
  and two requests a second apart both replay and judge the same window.
* **The background task is anchored** in a module-level set. The event loop keeps
  only a weak reference, so an unanchored run can be collected mid-flight and
  leave its row stuck at ``running``.
* **Phase 2 runs before phase 3.** A stale topic centroid must fail the run before
  a single replay or judge call is paid for.
* **The four phase outputs stay index-aligned** all the way into the persisted
  rows, and the counters describe the window rather than the sample.
* **Each row is attributed to its own session's agent**, not to the run's label:
  the default run is ``all``, and narrowing one flat table to one agent is the
  entire point of the column.
* **A failure marks the run ``failed``** with the reason in ``summary``, and
  re-raises so the caller still sees it.

No network: the embedding engine is a hand-written fake with explicit vectors,
the replay is stubbed, and the judge's ``LLMGateway`` call is patched. A run here
never touches an LLM provider or a live search.
"""

import asyncio
import gc
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunInFlightError,
    EmbeddingFingerprintMismatchError,
    InvalidCoverageParamsError,
    UnknownAgentLabelError,
)
from cognee.modules.recall_coverage.models import (
    RecallCoverageCuratedQuestion,
    RecallCoverageQuestion,
    RecallCoverageRun,
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)
from cognee.modules.recall_coverage.replay import ReplayedRow
from cognee.modules.recall_coverage.types import (
    AgentScope,
    AgentScopeMode,
    CoverageParams,
    QuestionSource,
    RunStatus,
)
from cognee.modules.search.operations.get_queries import QueryWindowRow

pipeline = importlib.import_module("cognee.modules.recall_coverage.pipeline")
repository = importlib.import_module("cognee.modules.recall_coverage.repository")
judge_module = importlib.import_module("cognee.modules.recall_coverage.judge")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

RUNBOOKS = "Where are the runbooks?"
CREDENTIALS = "How do I rotate credentials?"
ESCALATION = "What is our escalation path out of hours?"

# Deliberately orthogonal so nothing merges by accident and every threshold
# assertion is decided by these numbers rather than by a provider.
VECTORS: dict[str, list[float]] = {
    RUNBOOKS: [1.0, 0.0, 0.0],
    CREDENTIALS: [0.0, 1.0, 0.0],
    ESCALATION: [0.0, 0.0, 1.0],
}

MODEL = "fake/embedding-model"
DIMENSIONS = 3


class _FakeEngine:
    """Deterministic embeddings, one batch of two, and a real fingerprint."""

    model = MODEL

    def __init__(self, vectors: dict[str, list[float]] = None) -> None:
        self.vectors = vectors if vectors is not None else VECTORS
        self.calls: list[list[str]] = []

    async def embed_text(self, texts):
        self.calls.append(list(texts))
        return [self.vectors.get(text, [0.0, 0.0, 0.0]) for text in texts]

    def get_vector_size(self) -> int:
        return DIMENSIONS

    def get_batch_size(self) -> int:
        return 2


def _config(**overrides) -> RecallCoverageConfig:
    return RecallCoverageConfig(_env_file=None, **overrides)


def _params(**overrides) -> CoverageParams:
    return CoverageParams.from_config(_config(), **overrides)


def _user(user_id=None, tenant_id=None) -> SimpleNamespace:
    return SimpleNamespace(id=user_id or uuid4(), tenant_id=tenant_id)


def _scope(label: str = "all", mode: AgentScopeMode = AgentScopeMode.ALL) -> AgentScope:
    return AgentScope(label=label, prefixes=(), mode=mode)


def _window_row(
    text, *, user_id, dataset_id=None, created_at=BASE_TIME, session_id=None
) -> QueryWindowRow:
    return QueryWindowRow(
        query_id=uuid4(),
        text=text,
        query_type="GRAPH_COMPLETION",
        user_id=user_id,
        dataset_id=dataset_id,
        created_at=created_at,
        session_id=session_id,
    )


@pytest_asyncio.fixture
async def coverage_engine(tmp_path, monkeypatch):
    """A SQLite engine holding all five recall-coverage tables."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="recall_coverage_pipeline_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.create_all,
            tables=[
                RecallCoverageRun.__table__,
                RecallCoverageQuestion.__table__,
                RecallCoverageTopic.__table__,
                RecallCoverageTopicSuggestion.__table__,
                RecallCoverageCuratedQuestion.__table__,
            ],
        )

    monkeypatch.setattr(repository, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Wire the pipeline's collaborators to fakes. Returns the knobs to adjust.

    Everything expensive is a seam: the window query, the user load, the dataset
    resolution, the embedding engine and the replay. The judge is left real and
    only its ``LLMGateway`` call is patched, so the actual short-circuits (empty
    context scores 0 with no call; a 0 makes no completion call) still run.
    """
    state = SimpleNamespace(
        rows=[],
        datasets=[],
        engine=_FakeEngine(),
        contexts={},
        user=None,
        replay_calls=[],
    )

    async def fake_get_queries(**kwargs):
        state.query_kwargs = kwargs
        return list(state.rows)

    async def fake_get_user(user_id):
        return state.user if state.user is not None else _user(user_id)

    async def fake_datasets(datasets=None, permission_type="read", user=None):
        return list(state.datasets)

    async def fake_replay(questions, *, params, user_cache=None, search=None):
        state.replay_calls.append([question.text for question in questions])
        return [
            ReplayedRow(
                retrieval_context=state.contexts.get(question.text),
                dataset_name=None,
                payload_count=1 if state.contexts.get(question.text) else 0,
            )
            for question in questions
        ]

    monkeypatch.setattr(pipeline, "get_queries", fake_get_queries)
    monkeypatch.setattr(pipeline, "get_user", fake_get_user)
    monkeypatch.setattr(pipeline, "get_authorized_existing_datasets", fake_datasets)
    monkeypatch.setattr(pipeline, "get_embedding_engine", lambda: state.engine)
    monkeypatch.setattr(pipeline, "replay_questions", fake_replay)

    return state


def _llm_response(*, score=4):
    """One structured-output stand-in that satisfies both judge calls."""
    return SimpleNamespace(
        score=score,
        reason="The context names the runbook location.",
        answer="The runbooks live in the infra-docs repository.",
    )


async def _run(user, scope, params, config=None):
    run = await repository.create_run(user.id, scope.label, params=params)
    return await pipeline.run_recall_coverage(
        run.id, scope, user.id, params=params, config=config or _config()
    )


async def _topic(owner_id, label, centroid, *, model=MODEL):
    """A topic minted the way ``POST /topics`` mints one, centroid supplied."""
    return await repository.create_topic(
        owner_id,
        label,
        centroid=centroid,
        embedding_model=model,
        embedding_dimensions=DIMENSIONS,
    )


# --- the empty window --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_valid_label_with_no_traffic_completes_as_an_empty_run(
    coverage_engine, stub_pipeline
):
    """An agent that asked nothing in the window is not a failure."""
    user = _user()
    stub_pipeline.user = user
    params = _params()

    completed = await _run(user, _scope("claude-code", AgentScopeMode.PREFIX), params)

    assert completed.status == RunStatus.COMPLETE.value
    assert completed.summary["memory_score"] is None
    assert completed.summary["topics"] == []
    assert completed.summary["suggested_topics"] == []
    assert completed.question_count == 0
    assert completed.recall_count == 0
    assert await repository.load_run_questions(completed.id) == []
    # Nothing was embedded and nothing was replayed: an empty window costs nothing.
    assert stub_pipeline.engine.calls == []
    assert stub_pipeline.replay_calls == []


@pytest.mark.asyncio
async def test_the_window_is_the_configured_age_and_query_types(coverage_engine, stub_pipeline):
    user = _user()
    stub_pipeline.user = user
    params = _params(max_age_days=7)

    before = datetime.now(timezone.utc) - timedelta(days=7)
    await _run(user, _scope(), params)
    after = datetime.now(timezone.utc) - timedelta(days=7)

    kwargs = stub_pipeline.query_kwargs
    assert before <= kwargs["since"] <= after
    assert kwargs["query_types"] == params.query_types
    assert kwargs["session_scope"].label == "all"
    # No single-user filter — a run covers the caller's whole visible scope —
    # but that scope is bounded: a tenantless caller sees only themselves.
    assert "user_id" not in kwargs
    assert kwargs["user_ids"] == (stub_pipeline.user.id,)
    # The fetch is bounded in SQL; when the cap is hit, recall_count is corrected
    # by a COUNT(*) over the same filters.
    assert kwargs["limit"] == params.window_row_cap


# --- the full path -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_persists_index_aligned_judged_rows_and_window_counters(
    coverage_engine, stub_pipeline
):
    anna, ben = uuid4(), uuid4()
    caller = _user(anna)
    stub_pipeline.user = caller
    params = _params(min_scored_questions_per_topic=1)

    # Anna asked twice about runbooks inside the retry cooldown (one ask), plus
    # credentials once; Ben asked about escalation.
    stub_pipeline.rows = [
        _window_row(RUNBOOKS, user_id=anna, created_at=BASE_TIME),
        _window_row(RUNBOOKS, user_id=anna, created_at=BASE_TIME - timedelta(seconds=30)),
        _window_row(CREDENTIALS, user_id=anna, created_at=BASE_TIME - timedelta(hours=1)),
        _window_row(ESCALATION, user_id=ben, created_at=BASE_TIME - timedelta(hours=2)),
    ]
    # Only two of the three questions retrieved anything.
    stub_pipeline.contexts = {
        RUNBOOKS: "The runbooks live in infra-docs.",
        CREDENTIALS: "Credentials are rotated via the vault CLI.",
    }

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=_llm_response(score=4),
    ):
        completed = await _run(caller, _scope(), params)

    assert completed.status == RunStatus.COMPLETE.value
    # Four raw rows in the window; three question rows judged. The gap is the
    # retry the cooldown swallowed, and it is the only place that is visible.
    assert completed.recall_count == 4
    assert completed.question_count == 3

    rows = {row.question: row for row in await repository.load_run_questions(completed.id)}
    assert set(rows) == {RUNBOOKS, CREDENTIALS, ESCALATION}

    # The row that retrieved nothing scores 0 with no LLM call, and 0 means no
    # answer — the alignment is what makes this assertion meaningful at all.
    assert rows[ESCALATION].coverage_score == 0
    assert rows[ESCALATION].answer is None
    assert rows[RUNBOOKS].coverage_score == 4
    assert rows[RUNBOOKS].answer == "The runbooks live in the infra-docs repository."
    # The second runbooks row was a retry inside the cooldown: one ask, not two.
    # An agent looping on a question it cannot answer is not twice the demand.
    assert rows[RUNBOOKS].relevance == 1
    assert rows[RUNBOOKS].retrieval_context == "The runbooks live in infra-docs."
    assert rows[CREDENTIALS].relevance == 1
    # Two users' questions in one run: a run is tenant-wide and the UI filters.
    assert {rows[ESCALATION].user_id, rows[RUNBOOKS].user_id} == {anna, ben}


@pytest.mark.asyncio
async def test_curated_questions_join_the_same_run_without_displacing_traffic(
    coverage_engine, stub_pipeline
):
    caller = _user()
    stub_pipeline.user = caller
    dataset_id = uuid4()
    stub_pipeline.datasets = [SimpleNamespace(id=dataset_id, name="infra-docs")]
    params = _params(max_questions=1)

    stub_pipeline.rows = [_window_row(RUNBOOKS, user_id=caller.id, dataset_id=dataset_id)]
    stub_pipeline.contexts = {RUNBOOKS: "The runbooks live in infra-docs."}

    await repository.create_curated_question(caller, ESCALATION, config=_config())

    with patch.object(
        judge_module.LLMGateway,
        "acreate_structured_output",
        new_callable=AsyncMock,
        return_value=_llm_response(score=3),
    ):
        completed = await _run(caller, _scope(), params)

    rows = {row.question: row for row in await repository.load_run_questions(completed.id)}

    # max_questions=1 kept the one observed ask, and the user-defined question was
    # appended after the truncation rather than instead of it.
    assert set(rows) == {RUNBOOKS, ESCALATION}
    assert completed.question_count == 2
    assert rows[ESCALATION].source == QuestionSource.USER_DEFINED.value
    # Nobody asked it, so it carries no demand and no agent...
    assert rows[ESCALATION].relevance == 0
    assert rows[ESCALATION].agent_label is None
    assert rows[ESCALATION].dataset_id == dataset_id
    # ...and it feeds no average: only the observed row is counted.
    assert [topic["question_count"] for topic in completed.summary["topics"]] == [1]


@pytest.mark.asyncio
async def test_the_dataset_name_the_caller_can_see_is_written_onto_the_row(
    coverage_engine, stub_pipeline
):
    caller = _user()
    stub_pipeline.user = caller
    dataset_id = uuid4()
    stub_pipeline.datasets = [SimpleNamespace(id=dataset_id, name="infra-docs")]
    stub_pipeline.rows = [_window_row(RUNBOOKS, user_id=caller.id, dataset_id=dataset_id)]

    completed = await _run(caller, _scope(), _params())

    row = (await repository.load_run_questions(completed.id))[0]
    assert row.dataset_id == dataset_id
    assert row.dataset_name == "infra-docs"


# --- per-row agent attribution ------------------------------------------------


@pytest.mark.asyncio
async def test_an_all_run_attributes_each_row_to_its_own_sessions_agent(
    coverage_engine, stub_pipeline
):
    """The whole point of the column: one flat table the UI narrows to one agent.

    The run's own label is ``all``, so it says nothing about who asked what — only
    the row's own ``session_id`` does. A row with no session at all is ``api``, and
    that is the definition rather than a fallback.
    """
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [
        _window_row(RUNBOOKS, user_id=caller.id, session_id="claude_a1"),
        _window_row(
            CREDENTIALS,
            user_id=caller.id,
            created_at=BASE_TIME - timedelta(hours=1),
            session_id="codex_a1",
        ),
        _window_row(
            ESCALATION,
            user_id=caller.id,
            created_at=BASE_TIME - timedelta(hours=2),
            session_id=None,
        ),
    ]

    completed = await _run(caller, _scope(), _params())

    rows = {
        row.question: row.agent_label for row in await repository.load_run_questions(completed.id)
    }

    assert completed.agent_label == "all"
    assert rows == {RUNBOOKS: "claude-code", CREDENTIALS: "codex", ESCALATION: "api"}


@pytest.mark.asyncio
async def test_the_label_follows_the_deployments_prefix_map(coverage_engine, stub_pipeline):
    """Resolved through the config the run executed under, then stored.

    Re-deriving it at read time would relabel history the moment a deployment
    edits ``RECALL_COVERAGE_AGENT_PREFIX_MAP``; the label is frozen onto the row
    for the same reason ``params`` and ``summary`` are.
    """
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [_window_row(RUNBOOKS, user_id=caller.id, session_id="widget_7")]
    config = _config(agent_prefix_map='{"widget": ["widget_"]}')

    completed = await _run(caller, _scope(), _params(), config=config)

    assert (await repository.load_run_questions(completed.id))[0].agent_label == "widget"


@pytest.mark.asyncio
async def test_the_ui_label_is_resolved_like_any_other_agent(coverage_engine, stub_pipeline):
    """``ui`` is a human in the cloud search box, distinguishable only by its prefix."""
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [
        _window_row(RUNBOOKS, user_id=caller.id, session_id="search-ui-1712345678901"),
        _window_row(
            CREDENTIALS,
            user_id=caller.id,
            created_at=BASE_TIME - timedelta(hours=1),
            session_id="ui_abc",
        ),
    ]

    completed = await _run(caller, _scope(), _params())

    labels = {row.agent_label for row in await repository.load_run_questions(completed.id)}
    assert labels == {"ui"}


@pytest.mark.asyncio
async def test_a_labelled_run_still_reads_the_label_off_each_row(coverage_engine, stub_pipeline):
    """Not from ``scope.label``: a narrowed window happens to agree, and must not assume it.

    The window's own predicate is what restricts the rows; if the label were taken
    from the scope, the ``all`` run above would file every row under ``all``.
    """
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [_window_row(RUNBOOKS, user_id=caller.id, session_id="claude_a1")]

    completed = await _run(caller, _scope("claude-code", AgentScopeMode.PREFIX), _params())

    assert completed.agent_label == "claude-code"
    assert (await repository.load_run_questions(completed.id))[0].agent_label == "claude-code"


# --- phase 2 before phase 3 --------------------------------------------------


@pytest.mark.asyncio
async def test_a_stale_topic_centroid_fails_the_run_before_any_replay(
    coverage_engine, stub_pipeline
):
    """The fingerprint guard must fire before the only expensive phase."""
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [_window_row(RUNBOOKS, user_id=caller.id)]

    await _topic(caller.id, "Runbooks", (1.0, 0.0, 0.0), model="some/older-model")

    with pytest.raises(EmbeddingFingerprintMismatchError):
        await _run(caller, _scope(), _params())

    # Nothing was replayed, and the run is marked failed with the reason.
    assert stub_pipeline.replay_calls == []
    failed = (await repository.list_runs([caller.id]))[0]
    assert failed.status == RunStatus.FAILED.value
    assert "embedding" in failed.summary["error"].lower()


@pytest.mark.asyncio
async def test_questions_are_assigned_to_the_owners_topic_or_to_the_sink(
    coverage_engine, stub_pipeline
):
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [
        _window_row(RUNBOOKS, user_id=caller.id),
        _window_row(ESCALATION, user_id=caller.id),
    ]

    topic = await _topic(caller.id, "Runbooks", (1.0, 0.0, 0.0))

    completed = await _run(caller, _scope(), _params())

    rows = {row.question: row for row in await repository.load_run_questions(completed.id)}
    assert rows[RUNBOOKS].topic_id == topic.id
    # Orthogonal to the only topic, so the sink — stored as NULL, and reported as
    # a topic_id: null member of topics[] labelled "Uncategorized".
    assert rows[ESCALATION].topic_id is None
    reported = {cell["topic_id"]: cell["topic"] for cell in completed.summary["topics"]}
    assert reported == {str(topic.id): "Runbooks", None: "Uncategorized"}


@pytest.mark.asyncio
async def test_a_dense_sink_cluster_becomes_a_pending_suggestion(coverage_engine, stub_pipeline):
    caller = _user()
    stub_pipeline.user = caller
    # Three near-identical-but-distinct questions about one unmatched theme.
    texts = [f"How do I rotate the {name} credentials?" for name in ("vault", "s3", "db")]
    stub_pipeline.engine = _FakeEngine(
        {
            texts[0]: [1.0, 0.0, 0.0],
            texts[1]: [0.95, 0.31, 0.0],
            texts[2]: [0.95, 0.0, 0.31],
        }
    )
    stub_pipeline.rows = [
        _window_row(text, user_id=caller.id, created_at=BASE_TIME - timedelta(hours=index))
        for index, text in enumerate(texts)
    ]
    params = _params(min_questions_per_topic=2, dedup_threshold=0.99, sink_cluster_threshold=0.8)

    with patch(
        "cognee.modules.recall_coverage.suggest.generate_topic_label",
        new_callable=AsyncMock,
        return_value="Credential rotation",
    ):
        completed = await _run(caller, _scope(), params)

    pending = await repository.load_pending_suggestions(caller.id)
    assert [suggestion.label for suggestion in pending] == ["Credential rotation"]
    assert pending[0].question_count == 3
    assert pending[0].run_id == completed.id
    assert pending[0].agent_label == "all"
    # The suggestion is frozen into this run's report: the review queue moves as
    # the owner accepts and dismisses, and a historical run must keep showing what
    # it actually proposed. ``cohesion`` is not in it — it orders candidates and
    # says nothing about memory. The id is, because the dismiss route takes it and
    # the report is the only place it is published.
    assert completed.summary["suggested_topics"] == [
        {
            "suggestion_id": str(pending[0].id),
            "label": "Credential rotation",
            "question_count": 3,
        }
    ]
    # Every row went to the sink, which is one member row of topics[] — it reports
    # its own average (nothing was retrieved for any of them, so 0.0) and is still
    # excluded from the headline, because it is the absence of a topic.
    assert completed.summary["topics"] == [
        {
            "topic_id": None,
            "topic": "Uncategorized",
            "question_count": 3,
            "memory_score": 0.0,
        }
    ]
    assert completed.summary["memory_score"] is None


# --- starting a run ----------------------------------------------------------


@pytest.mark.asyncio
async def test_start_writes_a_pending_row_before_scheduling_anything(coverage_engine, monkeypatch):
    caller = _user()
    scheduled: list[tuple] = []

    monkeypatch.setattr(
        pipeline,
        "schedule_recall_coverage_run",
        lambda run_id, scope, user_id, *, params, config=None: scheduled.append(
            (run_id, scope.label, user_id)
        ),
    )

    run = await pipeline.start_recall_coverage_run(caller, None, config=_config())

    assert run.status == RunStatus.PENDING.value
    # Defaults to "all" when the request omits a label.
    assert run.agent_label == "all"
    assert run.params["max_questions"] == _config().max_questions
    # The row exists before the coroutine does, which is what the guard reads.
    assert scheduled == [(run.id, "all", caller.id)]
    assert [record.id for record in await repository.runs_in_flight(caller.id, "all")] == [run.id]


@pytest.mark.asyncio
async def test_a_second_run_for_the_same_owner_and_label_is_a_conflict(
    coverage_engine, monkeypatch
):
    caller = _user()
    monkeypatch.setattr(pipeline, "schedule_recall_coverage_run", lambda *args, **kwargs: None)

    await pipeline.start_recall_coverage_run(caller, "codex", config=_config())

    with pytest.raises(CoverageRunInFlightError):
        await pipeline.start_recall_coverage_run(caller, "codex", config=_config())

    # A different label, and a different owner, are both unaffected.
    await pipeline.start_recall_coverage_run(caller, "claude-code", config=_config())
    await pipeline.start_recall_coverage_run(_user(), "codex", config=_config())


@pytest.mark.asyncio
async def test_an_unknown_label_never_reaches_a_run_row(coverage_engine, monkeypatch):
    caller = _user()
    monkeypatch.setattr(pipeline, "schedule_recall_coverage_run", lambda *a, **k: None)

    with pytest.raises(UnknownAgentLabelError):
        await pipeline.start_recall_coverage_run(caller, "claude-codex", config=_config())

    assert await repository.list_runs([caller.id]) == []


@pytest.mark.asyncio
async def test_an_unknown_parameter_is_rejected_rather_than_ignored(coverage_engine, monkeypatch):
    """A run that looked accepted while using a different threshold is worse."""
    caller = _user()
    monkeypatch.setattr(pipeline, "schedule_recall_coverage_run", lambda *a, **k: None)

    with pytest.raises(InvalidCoverageParamsError):
        await pipeline.start_recall_coverage_run(
            caller, None, params={"max_question": 3}, config=_config()
        )

    with pytest.raises(InvalidCoverageParamsError):
        await pipeline.start_recall_coverage_run(
            caller, None, params={"dedup_threshold": 4.2}, config=_config()
        )

    assert await repository.list_runs([caller.id]) == []


@pytest.mark.asyncio
async def test_an_override_above_its_ceiling_is_rejected(coverage_engine, monkeypatch):
    """The field bounds stop meaningless runs; the ceilings stop unaffordable ones.

    ``POST /api/v1/coverage`` takes per-run overrides, so without a ceiling any authenticated
    caller can ask for a 200000-row window — a 200000 x 200000 similarity matrix
    before an LLM call — or a semaphore of 50000, and the ``(owner, agent_label)``
    guard bounds neither labels nor owners.
    """
    caller = _user()
    monkeypatch.setattr(pipeline, "schedule_recall_coverage_run", lambda *a, **k: None)
    ceilings = _config().override_ceilings()

    for name, ceiling in ceilings.items():
        with pytest.raises(InvalidCoverageParamsError):
            await pipeline.start_recall_coverage_run(
                caller, None, params={name: ceiling + 1}, config=_config()
            )

    assert await repository.list_runs([caller.id]) == []

    # At the ceiling is a legitimate, if large, run.
    run = await pipeline.start_recall_coverage_run(
        caller,
        None,
        params={"max_questions": ceilings["max_questions"]},
        config=_config(),
    )
    assert run.params["max_questions"] == ceilings["max_questions"]


def test_a_deployment_default_above_a_ceiling_is_the_operators_business():
    """Ceilings bound requests, not the deployment that set the defaults."""
    generous = RecallCoverageConfig(_env_file=None, max_questions=100_000)

    assert pipeline.build_params(None, generous).max_questions == 100_000


@pytest.mark.asyncio
async def test_overrides_are_snapshotted_onto_the_run_row(coverage_engine, monkeypatch):
    caller = _user()
    monkeypatch.setattr(pipeline, "schedule_recall_coverage_run", lambda *a, **k: None)

    run = await pipeline.start_recall_coverage_run(
        caller, None, params={"max_questions": 7, "judge_score_max": 4}, config=_config()
    )

    assert run.params["max_questions"] == 7
    # A run's scores are only comparable to another's when both snapshotted the
    # same scale, which is exactly why the scale is snapshotted.
    assert run.params["judge_score_max"] == 4


# --- the background task -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_background_run_is_anchored_until_it_finishes(coverage_engine, monkeypatch):
    """Without the anchor the loop's weak reference lets gc stop a run mid-flight."""
    release = asyncio.Event()
    caller = _user()

    async def fake_run(run_id, scope, user_id, *, params, config=None):
        await release.wait()

    monkeypatch.setattr(pipeline, "run_recall_coverage", fake_run)

    run = await pipeline.start_recall_coverage_run(caller, None, config=_config())

    assert len(pipeline._BACKGROUND_RUN_TASKS) == 1
    task = next(iter(pipeline._BACKGROUND_RUN_TASKS))

    gc.collect()
    assert task in pipeline._BACKGROUND_RUN_TASKS

    release.set()
    await task
    await asyncio.sleep(0)

    assert task not in pipeline._BACKGROUND_RUN_TASKS
    assert run.status == RunStatus.PENDING.value


# --- failure ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_phase_failure_marks_the_run_failed_and_re_raises(coverage_engine, stub_pipeline):
    caller = _user()
    stub_pipeline.user = caller

    async def explode(**kwargs):
        raise RuntimeError("the relational database went away")

    with patch.object(pipeline, "get_queries", explode):
        with pytest.raises(RuntimeError):
            await _run(caller, _scope(), _params())

    failed = (await repository.list_runs([caller.id]))[0]
    assert failed.status == RunStatus.FAILED.value
    # Class-prefixed and bounded, because this string is what RunInfo.error returns.
    assert failed.summary == {"error": "RuntimeError: the relational database went away"}
    assert failed.finished_at is not None


@pytest.mark.asyncio
async def test_a_row_whose_replay_failed_is_persisted_unjudged(coverage_engine, stub_pipeline):
    """One unreadable dataset must not throw away the rest of the run."""
    caller = _user()
    stub_pipeline.user = caller
    stub_pipeline.rows = [
        _window_row(RUNBOOKS, user_id=caller.id),
        _window_row(CREDENTIALS, user_id=caller.id, created_at=BASE_TIME - timedelta(hours=1)),
    ]
    stub_pipeline.contexts = {CREDENTIALS: "Credentials are rotated via the vault CLI."}

    async def fake_replay(questions, *, params, user_cache=None, search=None):
        return [
            ReplayedRow(
                retrieval_context=None,
                dataset_name=None,
                payload_count=0,
                error="dataset is not readable",
            )
            if question.text == RUNBOOKS
            else ReplayedRow(
                retrieval_context=stub_pipeline.contexts[question.text],
                dataset_name=None,
                payload_count=1,
            )
            for question in questions
        ]

    with patch.object(pipeline, "replay_questions", fake_replay):
        with patch.object(
            judge_module.LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=_llm_response(score=5),
        ):
            completed = await _run(caller, _scope(), _params())

    assert completed.status == RunStatus.COMPLETE.value
    rows = {row.question: row for row in await repository.load_run_questions(completed.id)}

    assert rows[RUNBOOKS].coverage_score is None
    assert rows[RUNBOOKS].error == "dataset is not readable"
    assert rows[CREDENTIALS].coverage_score == 5
