"""Guards on the recall-coverage HTTP surface — spec section 5, all twelve routes.

The load-bearing one is the **wire casing**. Response models here are plain
``pydantic.BaseModel`` and must stay that way: ``OutDTO`` sets
``alias_generator=to_camel`` (``cognee/api/DTO.py``) and FastAPI serializes
response models *by alias*, so switching a model's base class — a one-word change
that reads like tidying — would silently turn every ``topic_id`` into ``topicId``
and every ``judge_score`` into ``judgeScore``, breaking every client at once. No
other test in this repository protects that contract, so the assertions below are
deliberately literal.

Also pinned:

* an unknown ``agent_label`` is **404**, while a valid label with no traffic is an
  empty run — a typo must not be indistinguishable from "nothing asked yet";
* ``GET /agents`` reports labels discovered from traffic, never from a registry;
* every ``questions[]`` row carries ``source``, ``user_id`` and ``dataset_id``;
* id-keyed routes 404 on an owner mismatch and never 403, which would confirm
  that another owner's row with that id exists;
* the sink is the wire literal ``"other"`` and deleting it is a 422, not a 404;
* ``POST /runs`` is 202 and always background, 409 when one is already in flight;
  ``POST /curated-questions`` is 201; ``DELETE /curated-questions/{id}`` is 204.

Synchronous ``TestClient`` over a bare app with only this router mounted, plus the
``CogneeApiError`` handler the real app registers in ``cognee/api/client.py``. No
database, no LLM, no network: every repository call is a fake.
"""

from datetime import datetime, timezone
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError
from cognee.modules.recall_coverage.aggregate import CoverageRow, row_impact, summarize
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunInFlightError,
    CoverageRunNotFoundError,
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
)
from cognee.modules.recall_coverage.repository import (
    BenchmarkCell,
    CuratedQuestion,
    QuestionRecord,
    RunRecord,
    SuggestionRecord,
    TopicRecord,
)
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_ID,
    SINK_TOPIC_LABEL,
    CoverageParams,
    CuratedScope,
    QuestionSource,
    RunStatus,
    SuggestionStatus,
)

router_module = import_module("cognee.api.v1.recall_coverage.routers.get_recall_coverage_router")

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

OWNER_ID = uuid4()
OTHER_USER_ID = uuid4()
DATASET_ID = uuid4()
TOPIC_ID = uuid4()
RUN_ID = uuid4()

AGENT_LABEL = "claude-code"


def _config() -> RecallCoverageConfig:
    return RecallCoverageConfig(_env_file=None)


def _params(**overrides) -> CoverageParams:
    return CoverageParams.from_config(_config(), **overrides)


def _row(
    *,
    text="Where are the runbooks?",
    user_id=OWNER_ID,
    dataset_id=DATASET_ID,
    topic_id=TOPIC_ID,
    topic_label="Runbooks",
    judge_score=4,
    source=QuestionSource.OBSERVED.value,
) -> CoverageRow:
    return CoverageRow(
        question_text=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name="infra-docs" if dataset_id else None,
        question_group_id=uuid4(),
        source=source,
        was_asked=source == QuestionSource.OBSERVED.value,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        topic_id=topic_id,
        topic_label=topic_label if topic_id else SINK_TOPIC_LABEL,
        answer="They live in infra-docs.",
        judge_score=judge_score,
        judge_answered=None if judge_score is None else judge_score > 0,
        retrieval_context="Some context.",
        error=None,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        occurrence_count=3,
        impact=row_impact(3, judge_score, 5),
    )


def _record(row: CoverageRow) -> QuestionRecord:
    """The read-side twin of a persisted row."""
    return QuestionRecord(
        id=uuid4(),
        run_id=RUN_ID,
        question_group_id=row.question_group_id,
        user_id=row.user_id,
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        question_text=row.question_text,
        source=row.source,
        was_asked=row.was_asked,
        curated_question_id=row.curated_question_id,
        answer=row.answer,
        judge_score=row.judge_score,
        judge_answered=row.judge_answered,
        retrieval_context=row.retrieval_context,
        error=row.error,
        topic_id=row.topic_id,
        first_asked_at=row.first_asked_at,
        last_asked_at=row.last_asked_at,
        occurrence_count=row.occurrence_count,
        impact=row.impact,
    )


def _complete_run(rows, *, run_id=RUN_ID, agent_label=AGENT_LABEL) -> RunRecord:
    params = _params(min_scored_questions_per_topic=1)
    summary = summarize(rows, params=params, distinct_ask_count=9)
    return RunRecord(
        id=run_id,
        agent_label=agent_label,
        owner_id=OWNER_ID,
        status=RunStatus.COMPLETE.value,
        params=params.model_dump(mode="json"),
        summary=summary.to_dict(),
        finished_at=BASE_TIME,
        recall_row_count=14,
        distinct_ask_count=9,
        collapsed_retry_count=5,
        question_row_count=len(rows),
        curated_question_count=len([row for row in rows if not row.is_observed]),
        topic_count=1,
        dataset_count=1,
        user_count=2,
        taxonomy_version=4,
        created_at=BASE_TIME,
    )


def _pending_run(agent_label=AGENT_LABEL) -> RunRecord:
    return RunRecord(
        id=RUN_ID,
        agent_label=agent_label,
        owner_id=OWNER_ID,
        status=RunStatus.PENDING.value,
        params=_params().model_dump(mode="json"),
        summary=None,
        finished_at=None,
        created_at=BASE_TIME,
        taxonomy_version=4,
    )


def _topic(topic_id=TOPIC_ID, label="Runbooks", deleted_at=None) -> TopicRecord:
    return TopicRecord(
        id=topic_id,
        owner_id=OWNER_ID,
        label=label,
        centroid=(1.0, 0.0, 0.0),
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=3,
        seed_question_count=6,
        taxonomy_version=4,
        deleted_at=deleted_at,
        created_at=BASE_TIME,
    )


def _suggestion(status=SuggestionStatus.PENDING.value) -> SuggestionRecord:
    return SuggestionRecord(
        id=uuid4(),
        owner_id=OWNER_ID,
        label="Credential rotation",
        centroid=(1.0, 0.0, 0.0),
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=3,
        question_count=6,
        cohesion=0.88,
        status=status,
        agent_label="codex",
        run_id=RUN_ID,
        created_at=BASE_TIME,
    )


def _curated(scope=CuratedScope.AGENT.value, agent_label=AGENT_LABEL) -> CuratedQuestion:
    return CuratedQuestion(
        id=uuid4(),
        owner_id=OWNER_ID,
        scope=scope,
        agent_label=agent_label,
        question_text="What is our escalation path out of hours?",
        created_at=BASE_TIME,
    )


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    """A bare app with only this router, the real error handler, and no ambient env."""
    application = FastAPI()
    application.include_router(
        router_module.get_recall_coverage_router(), prefix="/api/v1/recall-coverage"
    )

    application.dependency_overrides[router_module.get_authenticated_user] = lambda: (
        SimpleNamespace(id=OWNER_ID, tenant_id=None)
    )

    # Registered by cognee/api/client.py in the real app. The routes here raise
    # CogneeApiError subclasses instead of hand-rolling a JSONResponse, so without
    # this the status codes under test would never be produced.
    @application.exception_handler(CogneeApiError)
    async def _handler(_, exc: CogneeApiError):
        return JSONResponse(
            status_code=exc.status_code, content={"detail": f"{exc.message} [{exc.name}]"}
        )

    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    # A developer's .env must not be able to move a threshold or a default limit.
    monkeypatch.setattr(router_module, "get_recall_coverage_config", _config)

    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# --- 3: the report, and the snake_case contract -------------------------------


@pytest.fixture
def report_client(client, monkeypatch) -> TestClient:
    rows = [
        _row(text="Where are the runbooks?", judge_score=4),
        _row(text="How do I rotate credentials?", judge_score=2),
        _row(text="What is our escalation path?", topic_id=None, judge_score=0),
        _row(
            text="Which alerts page whom?",
            source=QuestionSource.CURATED.value,
            dataset_id=None,
            topic_id=None,
            judge_score=1,
            user_id=OTHER_USER_ID,
        ),
    ]
    run = _complete_run(rows)
    records = [_record(row) for row in rows]

    async def fake_get_run(run_id, owner_ids):
        assert OWNER_ID in owner_ids
        if run_id != RUN_ID:
            raise CoverageRunNotFoundError()
        return run

    async def fake_questions(run_id):
        return records

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)
    return client


def test_the_report_is_snake_case_and_never_camel_case(report_client):
    """The one contract nothing else in this repo protects. Keep these literal."""
    response = report_client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}")
    assert response.status_code == 200
    body = response.json()

    assert "topic_id" in body["topics"][0]
    assert "topicId" not in body["topics"][0]

    # The same rule everywhere else in the payload.
    assert "overall_score" in body
    assert "overallScore" not in body
    assert "judge_score" in body["questions"][0]
    assert "judgeScore" not in body["questions"][0]
    assert "question_group_id" in body["questions"][0]
    assert "questionGroupId" not in body["questions"][0]
    assert "recall_row_count" in body["run"]
    assert "recallRowCount" not in body["run"]
    assert "dataset_name" in body["datasets"][0]
    assert "datasetName" not in body["datasets"][0]
    assert "user_id" in body["users"][0]
    assert "userId" not in body["users"][0]


def test_no_response_model_is_an_out_dto():
    """A one-word base-class change would camel-case the whole wire contract."""
    from cognee.api.DTO import OutDTO

    models = [
        value
        for name, value in vars(router_module).items()
        if isinstance(value, type) and issubclass(value, __import__("pydantic").BaseModel)
    ]
    assert models  # the sweep found something to check
    assert [model.__name__ for model in models if issubclass(model, OutDTO)] == []


def test_every_question_row_carries_its_source_user_and_dataset(report_client):
    body = report_client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()

    assert len(body["questions"]) == 4
    for question in body["questions"]:
        assert question["source"] in ("observed", "curated")
        assert question["user_id"]
        # Present on every row, and explicitly null for an unscoped one.
        assert "dataset_id" in question

    unscoped = [q for q in body["questions"] if q["dataset_id"] is None]
    assert unscoped and unscoped[0]["source"] == "curated"
    assert {q["user_id"] for q in body["questions"]} == {str(OWNER_ID), str(OTHER_USER_ID)}


def test_the_report_reads_the_frozen_summary_rather_than_recomputing_it(report_client):
    body = report_client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()

    # 4 and 2 on the one qualifying topic; the sink and the curated row are out.
    assert body["overall_score"] == 3.0
    assert body["topics"][0]["topic_id"] == str(TOPIC_ID)
    assert body["topics"][0]["label"] == "Runbooks"
    assert body["sink"]["topic_id"] == SINK_TOPIC_ID
    assert body["sink"]["label"] == SINK_TOPIC_LABEL
    assert body["sink"]["question_count"] == 1
    assert body["run"]["taxonomy_version"] == 4
    assert body["run"]["params"]["judge_score_max"] == 5


def test_a_sink_row_reports_the_wire_literal_other(report_client):
    body = report_client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()

    sink_rows = [q for q in body["questions"] if q["topic_id"] == SINK_TOPIC_ID]
    assert sink_rows
    assert {row["topic_label"] for row in sink_rows} == {SINK_TOPIC_LABEL}
    # The stored value is NULL; "other" only ever exists on the wire.
    assert all(row["topic_id"] != "None" for row in body["questions"])


def test_a_row_with_a_real_topic_is_never_labelled_other(client, monkeypatch):
    """The label map is the frozen summary, so a topic missing from it is unnameable.

    A topic whose only rows are curated has no observed rows to average, but it
    still received rows — and a response pairing a real ``topic_id`` with
    ``"Other"`` contradicts itself: one UI groups the row under Billing, another
    files it under a sink whose ``question_count`` is 0.
    """
    speculative = uuid4()
    rows = [
        _row(text="observed", judge_score=4),
        _row(
            text="curated only",
            source=QuestionSource.CURATED.value,
            topic_id=speculative,
            topic_label="Billing & invoices",
        ),
    ]
    run = _complete_run(rows)
    records = [_record(row) for row in rows]

    async def fake_get_run(run_id, owner_ids):
        return run

    async def fake_questions(run_id):
        return records

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)

    body = client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()
    curated = next(row for row in body["questions"] if row["question_text"] == "curated only")

    assert curated["topic_id"] == str(speculative)
    assert curated["topic_label"] == "Billing & invoices"
    # And the topic is in the frozen breakdown, with no observed rows to average.
    cell = next(cell for cell in body["topics"] if cell["topic_id"] == str(speculative))
    assert (cell["label"], cell["question_count"], cell["avg_score"]) == (
        "Billing & invoices",
        0,
        None,
    )


def test_a_run_in_another_owner_scope_is_404_not_403(report_client):
    response = report_client.get(f"/api/v1/recall-coverage/runs/{uuid4()}")
    assert response.status_code == 404
    assert "CoverageRunNotFoundError" in response.json()["detail"]


def test_sink_alerts_reach_the_wire_as_a_code_and_a_message(client, monkeypatch):
    """The frozen summary is splatted into the response models, so its shape matters.

    ``sink`` and every breakdown cell are built with ``Model(**cell)`` off the
    stored JSON. That is the one place a drift between what
    ``aggregate.summarize`` writes and what this router declares would surface as a
    500 on a perfectly good run rather than as a missing field, so the alert list —
    the most structured thing in the summary — is checked end to end.
    """
    # Two of three observed rows in the sink: 0.67, above the 0.30 alert share.
    rows = [
        _row(text="q1", topic_id=TOPIC_ID, judge_score=4),
        _row(text="q2", topic_id=None, judge_score=1),
        _row(text="q3", topic_id=None, judge_score=0),
    ]
    params = _params(min_scored_questions_per_topic=1)
    run = _complete_run(rows)
    run = RunRecord(
        **{
            **vars(run),
            "summary": summarize(
                rows, params=params, distinct_ask_count=3, sink_cluster_sizes=[12]
            ).to_dict(),
        }
    )

    async def fake_get_run(run_id, owner_ids):
        return run

    async def fake_questions(run_id):
        return []

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)

    sink = client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()["sink"]

    assert {alert["code"] for alert in sink["alerts"]} == {
        "sink_share_above_threshold",
        "large_sink_cluster",
    }
    assert all(alert["message"] for alert in sink["alerts"])
    assert sink["question_count"] == 2


def test_a_summary_from_a_newer_version_still_renders(client, monkeypatch):
    """Unknown keys are ignored rather than fatal, which is why the splat is safe.

    A run persisted by a later version of ``aggregate`` must still be readable —
    historical runs are the whole point of freezing the summary, so a new field
    must not make every older reader 500.
    """
    run = _complete_run([_row()])
    summary = {
        **run.summary,
        "a_field_from_the_future": 1,
        "sink": {**run.summary["sink"], "a_new_sink_field": True},
    }
    run = RunRecord(**{**vars(run), "summary": summary})

    async def fake_get_run(run_id, owner_ids):
        return run

    async def fake_questions(run_id):
        return []

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)

    response = client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}")

    assert response.status_code == 200
    assert "a_field_from_the_future" not in response.json()
    assert "a_new_sink_field" not in response.json()["sink"]


def test_a_pending_run_renders_as_no_numbers_yet(client, monkeypatch):
    async def fake_get_run(run_id, owner_ids):
        return _pending_run()

    async def fake_questions(run_id):
        return []

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)

    body = client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()

    assert body["run"]["status"] == "pending"
    assert body["overall_score"] is None
    assert body["topics"] == []
    assert body["questions"] == []
    assert body["sink"]["question_count"] == 0


def test_a_failed_run_does_not_pretend_to_have_breakdowns(client, monkeypatch):
    failed = RunRecord(
        id=RUN_ID,
        agent_label=AGENT_LABEL,
        owner_id=OWNER_ID,
        status=RunStatus.FAILED.value,
        params=None,
        summary={"error": "the embedding engine returned zero vectors"},
        finished_at=BASE_TIME,
        created_at=BASE_TIME,
    )

    async def fake_get_run(run_id, owner_ids):
        return failed

    async def fake_questions(run_id):
        return []

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)

    body = client.get(f"/api/v1/recall-coverage/runs/{RUN_ID}").json()

    assert body["run"]["status"] == "failed"
    assert body["overall_score"] is None
    assert body["datasets"] == []


# --- 1: starting a run --------------------------------------------------------


def test_starting_a_run_is_202_and_pending(client, monkeypatch):
    started: list[dict] = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        started.append({"label": agent_label, "params": params, "user_id": user.id})
        return _pending_run()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    response = client.post(
        "/api/v1/recall-coverage/runs",
        json={"agent_label": AGENT_LABEL, "params": {"max_questions": 7}},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert response.json()["run_id"] == str(RUN_ID)
    assert started == [{"label": AGENT_LABEL, "params": {"max_questions": 7}, "user_id": OWNER_ID}]


def test_an_omitted_agent_label_defaults_to_all(client, monkeypatch):
    seen: list = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        seen.append(agent_label)
        # What resolve_agent_scope does with None, and what the row then carries.
        return _pending_run(agent_label="all")

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    # Both an empty body and no body at all mean "all".
    assert client.post("/api/v1/recall-coverage/runs", json={}).json()["agent_label"] == "all"
    assert client.post("/api/v1/recall-coverage/runs").json()["agent_label"] == "all"
    assert seen == [None, None]


def test_a_run_already_in_flight_is_409(client, monkeypatch):
    async def fake_start(user, agent_label=None, *, params=None, config=None):
        raise CoverageRunInFlightError()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    response = client.post("/api/v1/recall-coverage/runs", json={"agent_label": AGENT_LABEL})

    assert response.status_code == 409
    assert "CoverageRunInFlightError" in response.json()["detail"]


def test_the_request_body_accepts_camel_case_too(client, monkeypatch):
    """Requests use InDTO, which is the one place camelCase is welcome."""
    seen: list = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        seen.append(agent_label)
        return _pending_run()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    client.post("/api/v1/recall-coverage/runs", json={"agentLabel": AGENT_LABEL})
    assert seen == [AGENT_LABEL]


# --- 2: listing runs ----------------------------------------------------------


def test_listing_runs_applies_the_configured_default_limit(client, monkeypatch):
    calls: list[tuple] = []

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        calls.append((tuple(owner_ids), agent_label, limit))
        return [_complete_run([_row()])]

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    body = client.get("/api/v1/recall-coverage/runs").json()
    assert len(body) == 1
    assert body[0]["agent_label"] == AGENT_LABEL
    assert calls[0][1] is None
    assert calls[0][2] == _config().runs_list_default_limit

    client.get("/api/v1/recall-coverage/runs", params={"agent_label": AGENT_LABEL, "limit": 2})
    assert calls[1][1] == AGENT_LABEL
    assert calls[1][2] == 2


def test_an_unknown_agent_label_is_404_everywhere_it_is_accepted(client, monkeypatch):
    """A typo must not be indistinguishable from 'this agent asked nothing'."""

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        raise AssertionError("an unknown label must never reach the repository")

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    response = client.get("/api/v1/recall-coverage/runs", params={"agent_label": "claude-codex"})

    assert response.status_code == 404
    assert "UnknownAgentLabelError" in response.json()["detail"]


def test_a_valid_label_with_no_runs_is_an_empty_list_not_an_error(client, monkeypatch):
    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        return []

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    response = client.get("/api/v1/recall-coverage/runs", params={"agent_label": "codex"})
    assert response.status_code == 200
    assert response.json() == []


# --- 4: agents ----------------------------------------------------------------


def test_agents_are_discovered_from_traffic_and_joined_to_the_latest_run(client, monkeypatch):
    windows = [
        SimpleNamespace(label="all", recall_row_count=14),
        SimpleNamespace(label=AGENT_LABEL, recall_row_count=9),
    ]
    complete = _complete_run([_row()], agent_label=AGENT_LABEL)

    async def fake_counts(*, since=None, query_types=None, labels=None, config=None):
        assert since is not None
        assert query_types
        return windows

    async def fake_latest(owner_ids, agent_labels=None):
        assert set(agent_labels) == {"all", AGENT_LABEL}
        return {AGENT_LABEL: complete}

    monkeypatch.setattr(router_module, "agent_window_counts", fake_counts)
    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)

    body = client.get("/api/v1/recall-coverage/agents").json()

    # Busiest first, and only labels that asked something appear at all.
    assert [row["agent_label"] for row in body] == ["all", AGENT_LABEL]
    assert body[0]["recall_row_count"] == 14
    # A label with traffic but no finished run is shown with a null run, not hidden.
    assert body[0]["latest_run"] is None
    assert body[0]["overall_score"] is None
    assert body[1]["latest_run"]["run_id"] == str(RUN_ID)
    # Read off that run's frozen summary — one row scoring 4 on one topic.
    assert body[1]["overall_score"] == 4.0


def test_agents_defaults_to_the_top_five(client, monkeypatch):
    """ "I don't show 10, only show the top five."""
    windows = [
        SimpleNamespace(label=f"label-{index}", recall_row_count=100 - index) for index in range(9)
    ]

    async def fake_counts(**kwargs):
        return windows

    async def fake_latest(owner_ids, agent_labels=None):
        return {}

    monkeypatch.setattr(router_module, "agent_window_counts", fake_counts)
    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)

    assert _config().agents_list_default_limit == 5
    assert len(client.get("/api/v1/recall-coverage/agents").json()) == 5
    assert len(client.get("/api/v1/recall-coverage/agents", params={"limit": 2}).json()) == 2


def test_no_traffic_means_no_agents_rather_than_a_registry_dump(client, monkeypatch):
    async def fake_counts(**kwargs):
        return []

    async def fake_latest(owner_ids, agent_labels=None):
        raise AssertionError("nothing to join when no label has traffic")

    monkeypatch.setattr(router_module, "agent_window_counts", fake_counts)
    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)

    # The route must not even attempt the join — an empty label list would make
    # latest_complete_runs fall back to "every label" and scan for nothing.
    response = client.get("/api/v1/recall-coverage/agents")
    assert response.status_code == 200
    assert response.json() == []


# --- 5, 6, 7, 8: topics and suggestions ---------------------------------------


def test_topics_are_owner_scoped_and_carry_the_pending_suggestions(client, monkeypatch):
    suggestion = _suggestion()
    calls: list[bool] = []

    async def fake_topics(owner_ids, *, include_deleted=False):
        calls.append(include_deleted)
        return [_topic()] + (
            [_topic(uuid4(), "Gone", deleted_at=BASE_TIME)] if include_deleted else []
        )

    async def fake_pending(owner_id):
        return [suggestion]

    async def fake_version(owner_id):
        return 4

    monkeypatch.setattr(router_module, "list_topics", fake_topics)
    monkeypatch.setattr(router_module, "load_pending_suggestions", fake_pending)
    monkeypatch.setattr(router_module, "current_taxonomy_version", fake_version)

    body = client.get("/api/v1/recall-coverage/topics").json()

    assert body["taxonomy_version"] == 4
    assert [topic["topic_id"] for topic in body["topics"]] == [str(TOPIC_ID)]
    assert body["topics"][0]["deleted_at"] is None
    assert body["suggestions"][0]["suggestion_id"] == str(suggestion.id)
    assert body["suggestions"][0]["status"] == "pending"
    assert body["suggestions"][0]["cohesion"] == 0.88
    # There is no agent_label parameter: one taxonomy serves every agent.
    assert "agent_label" not in router_module.TopicsResponse.model_fields

    deleted_body = client.get(
        "/api/v1/recall-coverage/topics", params={"include_deleted": "true"}
    ).json()
    assert len(deleted_body["topics"]) == 2
    assert calls == [False, True]


def test_deleting_a_topic_returns_the_new_version(client, monkeypatch):
    async def fake_delete(topic_id, owner_ids):
        assert topic_id == TOPIC_ID
        assert OWNER_ID in owner_ids
        return 5

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    response = client.delete(f"/api/v1/recall-coverage/topics/{TOPIC_ID}")
    assert response.status_code == 200
    assert response.json() == {"taxonomy_version": 5}


def test_deleting_the_sink_is_422(client, monkeypatch):
    async def fake_delete(topic_id, owner_ids):
        raise AssertionError("the sink is not a row and must never reach the repository")

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    response = client.delete(f"/api/v1/recall-coverage/topics/{SINK_TOPIC_ID}")

    assert response.status_code == 422
    assert "SinkTopicNotEditableError" in response.json()["detail"]


def test_deleting_a_topic_in_another_owner_scope_is_404(client, monkeypatch):
    async def fake_delete(topic_id, owner_ids):
        raise CoverageTopicNotFoundError()

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    response = client.delete(f"/api/v1/recall-coverage/topics/{uuid4()}")
    assert response.status_code == 404
    assert "CoverageTopicNotFoundError" in response.json()["detail"]

    # A malformed id is the same 404, not a 422 telling the caller how ids look.
    assert client.delete("/api/v1/recall-coverage/topics/not-a-uuid").status_code == 404


def test_accepting_a_suggestion_returns_the_minted_topic(client, monkeypatch):
    suggestion = _suggestion()

    async def fake_accept(suggestion_id, owner_ids):
        assert suggestion_id == suggestion.id
        return _topic(label="Credential rotation"), _suggestion(
            status=SuggestionStatus.ACCEPTED.value
        )

    monkeypatch.setattr(router_module, "accept_topic_suggestion", fake_accept)

    response = client.post(f"/api/v1/recall-coverage/suggestions/{suggestion.id}/accept")

    assert response.status_code == 200
    assert response.json()["topic_id"] == str(TOPIC_ID)
    assert response.json()["label"] == "Credential rotation"
    assert response.json()["taxonomy_version"] == 4


def test_accepting_a_decided_suggestion_is_409_and_a_missing_one_404(client, monkeypatch):
    async def not_pending(suggestion_id, owner_ids):
        raise CoverageSuggestionNotPendingError()

    monkeypatch.setattr(router_module, "accept_topic_suggestion", not_pending)
    assert client.post(f"/api/v1/recall-coverage/suggestions/{uuid4()}/accept").status_code == 409

    async def missing(suggestion_id, owner_ids):
        raise CoverageSuggestionNotFoundError()

    monkeypatch.setattr(router_module, "accept_topic_suggestion", missing)
    assert client.post(f"/api/v1/recall-coverage/suggestions/{uuid4()}/accept").status_code == 404


def test_dismissing_a_suggestion_returns_it_settled(client, monkeypatch):
    async def fake_dismiss(suggestion_id, owner_ids):
        return _suggestion(status=SuggestionStatus.DISMISSED.value)

    monkeypatch.setattr(router_module, "dismiss_topic_suggestion", fake_dismiss)

    response = client.post(f"/api/v1/recall-coverage/suggestions/{uuid4()}/dismiss")

    assert response.status_code == 200
    assert response.json()["status"] == "dismissed"
    assert response.json()["accepted_topic_id"] is None


# --- 9, 10, 11: curated questions ---------------------------------------------


def test_adding_a_curated_question_is_201(client, monkeypatch):
    seen: list[tuple] = []

    async def fake_create(user, question_text, scope=None, agent_label=None, *, config=None):
        seen.append((question_text, scope, agent_label))
        return _curated()

    monkeypatch.setattr(router_module, "create_curated_question", fake_create)

    response = client.post(
        "/api/v1/recall-coverage/curated-questions",
        json={
            "question_text": "What is our escalation path out of hours?",
            "scope": "agent",
            "agent_label": AGENT_LABEL,
        },
    )

    assert response.status_code == 201
    assert response.json()["scope"] == "agent"
    assert response.json()["question_text"].startswith("What is our escalation")
    assert seen == [("What is our escalation path out of hours?", "agent", AGENT_LABEL)]


def test_a_duplicate_curated_question_is_409(client, monkeypatch):
    async def fake_create(user, question_text, scope=None, agent_label=None, *, config=None):
        raise DuplicateCuratedQuestionError()

    monkeypatch.setattr(router_module, "create_curated_question", fake_create)

    response = client.post(
        "/api/v1/recall-coverage/curated-questions",
        json={"question_text": "Where are the runbooks?", "scope": "shared"},
    )

    assert response.status_code == 409
    assert "DuplicateCuratedQuestionError" in response.json()["detail"]


def test_listing_curated_questions_returns_the_label_plus_the_shared_rows(client, monkeypatch):
    async def fake_list(user, agent_label=None, *, config=None):
        assert agent_label == AGENT_LABEL
        return [_curated(), _curated(scope=CuratedScope.SHARED.value, agent_label=None)]

    monkeypatch.setattr(router_module, "list_curated_questions", fake_list)

    body = client.get(
        "/api/v1/recall-coverage/curated-questions", params={"agent_label": AGENT_LABEL}
    ).json()

    assert [item["scope"] for item in body] == ["agent", "shared"]
    assert body[1]["agent_label"] is None


def test_deleting_a_curated_question_is_204_and_404_out_of_scope(client, monkeypatch):
    deleted: list = []

    async def fake_delete(user, question_id):
        deleted.append(question_id)

    monkeypatch.setattr(router_module, "delete_curated_question", fake_delete)

    question_id = uuid4()
    response = client.delete(f"/api/v1/recall-coverage/curated-questions/{question_id}")
    assert response.status_code == 204
    assert not response.content
    assert deleted == [question_id]

    async def fake_missing(user, question_id):
        raise CuratedQuestionNotFoundError()

    monkeypatch.setattr(router_module, "delete_curated_question", fake_missing)
    assert client.delete(f"/api/v1/recall-coverage/curated-questions/{uuid4()}").status_code == 404


# --- 12: the benchmark matrix -------------------------------------------------


def test_the_summary_matrix_is_datasets_by_agents_over_shared_curated_rows(client, monkeypatch):
    runs = {
        AGENT_LABEL: _complete_run([_row()], run_id=RUN_ID, agent_label=AGENT_LABEL),
        "codex": _complete_run([_row()], run_id=uuid4(), agent_label="codex"),
    }
    requested: list = []

    async def fake_latest(owner_ids, agent_labels=None):
        requested.append(agent_labels)
        return runs

    async def fake_cells(run_ids_by_label):
        assert set(run_ids_by_label) == {AGENT_LABEL, "codex"}
        return [
            BenchmarkCell(
                agent_label=AGENT_LABEL,
                run_id=RUN_ID,
                dataset_id=DATASET_ID,
                dataset_name="infra-docs",
                question_count=4,
                scored_question_count=4,
                avg_score=2.5,
            )
        ]

    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)
    monkeypatch.setattr(router_module, "benchmark_cells", fake_cells)

    body = client.get(
        "/api/v1/recall-coverage/summary", params={"agent_labels": f"{AGENT_LABEL}, codex"}
    ).json()

    assert requested == [[AGENT_LABEL, "codex"]]
    assert body["judge_score_max"] == 5
    assert body["agent_labels"] == sorted([AGENT_LABEL, "codex"])
    assert body["cells"][0]["dataset_name"] == "infra-docs"
    assert body["cells"][0]["avg_score"] == 2.5
    # 2.5 out of 5.
    assert body["cells"][0]["score_pct"] == 50.0
    assert body["cells"][0]["judge_score_max"] == 5
    assert "score_pct" in body["cells"][0]
    assert "scorePct" not in body["cells"][0]


def test_each_matrix_cell_is_a_percentage_of_its_own_runs_scale(client, monkeypatch):
    """A run judged 0..10 must not be restated against today's default of 5.

    ``params`` is frozen on the run row so a historical run stays readable after
    the deployment's defaults move; dividing its mean by the live default would
    report a cell averaging 8 out of 10 as 160%, and would put two labels whose
    runs used different scales on one axis.
    """
    ten_point = _complete_run([_row()], run_id=RUN_ID, agent_label=AGENT_LABEL)
    ten_point = RunRecord(**{**ten_point.__dict__, "params": {"judge_score_max": 10}})
    codex_run_id = uuid4()

    async def fake_latest(owner_ids, agent_labels=None):
        return {
            AGENT_LABEL: ten_point,
            "codex": _complete_run([_row()], run_id=codex_run_id, agent_label="codex"),
        }

    async def fake_cells(run_ids_by_label):
        return [
            BenchmarkCell(
                agent_label=AGENT_LABEL,
                run_id=RUN_ID,
                dataset_id=DATASET_ID,
                dataset_name="infra-docs",
                question_count=2,
                scored_question_count=2,
                avg_score=8.0,
            ),
            BenchmarkCell(
                agent_label="codex",
                run_id=codex_run_id,
                dataset_id=DATASET_ID,
                dataset_name="infra-docs",
                question_count=2,
                scored_question_count=2,
                avg_score=4.0,
            ),
        ]

    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)
    monkeypatch.setattr(router_module, "benchmark_cells", fake_cells)

    body = client.get("/api/v1/recall-coverage/summary").json()
    by_label = {cell["agent_label"]: cell for cell in body["cells"]}

    # 8 of 10, not 160% of 5.
    assert by_label[AGENT_LABEL]["judge_score_max"] == 10
    assert by_label[AGENT_LABEL]["score_pct"] == 80.0
    # The other label's run used the deployment default, and keeps it.
    assert by_label["codex"]["judge_score_max"] == 5
    assert by_label["codex"]["score_pct"] == 80.0
    # The matrix still reports what a run started now would use.
    assert body["judge_score_max"] == 5


@pytest.mark.parametrize("path", ["runs", "agents"])
def test_a_non_positive_limit_is_422_rather_than_a_negative_sql_limit(client, path):
    """``LIMIT -1`` is "no limit" on SQLite and an error on Postgres."""
    assert client.get(f"/api/v1/recall-coverage/{path}", params={"limit": -1}).status_code == 422
    assert client.get(f"/api/v1/recall-coverage/{path}", params={"limit": 0}).status_code == 422


def test_the_summary_matrix_404s_on_an_unknown_label(client, monkeypatch):
    async def fake_latest(owner_ids, agent_labels=None):
        raise AssertionError("an unknown label must never reach the repository")

    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)

    response = client.get(
        "/api/v1/recall-coverage/summary", params={"agent_labels": "claude-code,nope"}
    )
    assert response.status_code == 404


def test_the_summary_matrix_omitted_labels_mean_every_label_with_a_run(client, monkeypatch):
    requested: list = []

    async def fake_latest(owner_ids, agent_labels=None):
        requested.append(agent_labels)
        return {}

    async def fake_cells(run_ids_by_label):
        assert run_ids_by_label == {}
        return []

    monkeypatch.setattr(router_module, "latest_complete_runs", fake_latest)
    monkeypatch.setattr(router_module, "benchmark_cells", fake_cells)

    body = client.get("/api/v1/recall-coverage/summary").json()

    assert requested == [None]
    assert body["cells"] == []
    assert body["agent_labels"] == []


# --- OpenAPI ------------------------------------------------------------------


def test_every_route_of_the_spec_is_mounted(app):
    schema = app.openapi()
    operations = {
        (method.upper(), path.replace("/api/v1/recall-coverage", ""))
        for path, methods in schema["paths"].items()
        for method in methods
    }

    assert operations == {
        ("POST", "/runs"),
        ("GET", "/runs"),
        ("GET", "/runs/{run_id}"),
        ("GET", "/agents"),
        ("GET", "/topics"),
        ("DELETE", "/topics/{topic_id}"),
        ("POST", "/suggestions/{suggestion_id}/accept"),
        ("POST", "/suggestions/{suggestion_id}/dismiss"),
        ("POST", "/curated-questions"),
        ("GET", "/curated-questions"),
        ("DELETE", "/curated-questions/{question_id}"),
        ("GET", "/summary"),
    }


def test_the_declared_schema_is_snake_case(app):
    """A second, request-free guard: the OpenAPI components must not be camel."""
    components = app.openapi()["components"]["schemas"]

    for name in ("QuestionRow", "TopicCell", "RunInfo", "CoverageReport"):
        properties = components[name]["properties"]
        assert not [key for key in properties if key != key.lower()], (name, properties.keys())
