"""Guards on the coverage HTTP surface — all ten routes at ``/api/v1/coverage``.

The load-bearing one is the **wire casing**. Response models here are plain
``pydantic.BaseModel`` and must stay that way: ``OutDTO`` sets
``alias_generator=to_camel`` (``cognee/api/DTO.py``) and FastAPI serializes
response models *by alias*, so switching a model's base class — a one-word change
that reads like tidying — would silently turn every ``topic_id`` into ``topicId``
and every ``coverage_score`` into ``coverageScore``, breaking every client at once.
No other test in this repository protects that contract, so the assertions below
are deliberately literal.

Also pinned:

* an unknown ``agent_label`` is **422**, while a valid label with no traffic is an
  empty run — a typo must not be indistinguishable from "nothing asked yet";
* the report is read from the run's **frozen summary**, never recomputed, and a
  payload written by another version of ``aggregate`` still renders;
* each ``questions[]`` row carries its own ``agent``, resolved from that row's
  session and not from the run's label;
* ``answer`` is returned only on the caller's own rows;
* the sink is ``topic_id: null`` / ``"Uncategorized"`` inside ``topics[]``, and has
  no id to address it by;
* id-keyed routes 404 on an owner mismatch and never 403, which would confirm
  that another owner's row with that id exists;
* ``POST ""`` is 202 and always background, 409 when one is already in flight;
  ``POST /questions`` and ``POST /topics`` are 201; the three deletes and the
  dismiss are 204 with no body.

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
from cognee.modules.recall_coverage.aggregate import CoverageRow, SuggestedTopic, summarize
from cognee.modules.recall_coverage.config import RecallCoverageConfig
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunInFlightError,
    CoverageRunNotFoundError,
    CoverageSuggestionNotFoundError,
    CoverageSuggestionNotPendingError,
    CoverageTopicNotFoundError,
    CuratedQuestionLimitError,
    CuratedQuestionNotFoundError,
    DuplicateCuratedQuestionError,
    DuplicateTopicError,
    EmptyTopicLabelError,
)
from cognee.modules.recall_coverage.repository import (
    CuratedQuestion,
    QuestionRecord,
    RunRecord,
    SuggestionRecord,
    TopicRecord,
)
from cognee.modules.recall_coverage.types import (
    SINK_TOPIC_LABEL,
    CoverageParams,
    QuestionSource,
    RunStatus,
    SuggestionStatus,
)

router_module = import_module("cognee.api.v1.recall_coverage.routers.get_recall_coverage_router")

PREFIX = "/api/v1/coverage"

BASE_TIME = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)

OWNER_ID = uuid4()
OTHER_USER_ID = uuid4()
DATASET_ID = uuid4()
TOPIC_ID = uuid4()
RUN_ID = uuid4()
SUGGESTION_ID = uuid4()

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
    topic="Runbooks",
    coverage_score=4,
    relevance=3,
    agent_label=AGENT_LABEL,
    source=QuestionSource.OBSERVED.value,
) -> CoverageRow:
    return CoverageRow(
        question=text,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name="infra-docs" if dataset_id else None,
        agent_label=agent_label if relevance > 0 else None,
        source=source,
        curated_question_id=None if source == QuestionSource.OBSERVED.value else uuid4(),
        topic_id=topic_id,
        topic=topic if topic_id else SINK_TOPIC_LABEL,
        answer="They live in infra-docs.",
        coverage_score=coverage_score,
        retrieval_context="Some context.",
        error=None,
        first_asked_at=BASE_TIME,
        last_asked_at=BASE_TIME,
        relevance=relevance,
    )


def _record(row: CoverageRow) -> QuestionRecord:
    """The read-side twin of a persisted row.

    Note what is *not* on it: the topic label. The read endpoint resolves that from
    the frozen summary, so a topic deleted since the run still names what it
    measured.
    """
    return QuestionRecord(
        id=uuid4(),
        run_id=RUN_ID,
        user_id=row.user_id,
        dataset_id=row.dataset_id,
        dataset_name=row.dataset_name,
        question=row.question,
        agent_label=row.agent_label,
        source=row.source,
        curated_question_id=row.curated_question_id,
        answer=row.answer,
        coverage_score=row.coverage_score,
        retrieval_context=row.retrieval_context,
        error=row.error,
        topic_id=row.topic_id,
        first_asked_at=row.first_asked_at,
        last_asked_at=row.last_asked_at,
        relevance=row.relevance,
    )


def _complete_run(rows, *, run_id=RUN_ID, agent_label=AGENT_LABEL, suggested=()) -> RunRecord:
    params = _params(min_scored_questions_per_topic=1)
    summary = summarize(rows, params=params, suggested_topics=suggested)
    return RunRecord(
        id=run_id,
        agent_label=agent_label,
        owner_id=OWNER_ID,
        status=RunStatus.COMPLETE.value,
        params=params.model_dump(mode="json"),
        summary=summary.to_dict(),
        finished_at=BASE_TIME,
        recall_count=14,
        question_count=len(rows),
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
    )


def _failed_run(message="RuntimeError: the relational database went away") -> RunRecord:
    return RunRecord(
        id=RUN_ID,
        agent_label=AGENT_LABEL,
        owner_id=OWNER_ID,
        status=RunStatus.FAILED.value,
        params=_params().model_dump(mode="json"),
        summary={"error": message},
        finished_at=BASE_TIME,
        created_at=BASE_TIME,
    )


def _topic(topic_id=TOPIC_ID, label="Runbooks", deleted_at=None) -> TopicRecord:
    return TopicRecord(
        id=topic_id,
        owner_id=OWNER_ID,
        label=label,
        centroid=(1.0, 0.0, 0.0),
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=3,
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


def _curated() -> CuratedQuestion:
    return CuratedQuestion(
        id=uuid4(),
        owner_id=OWNER_ID,
        question="What is our escalation path out of hours?",
        created_at=BASE_TIME,
    )


@pytest.fixture
def app(monkeypatch) -> FastAPI:
    """A bare app with only this router, the real error handler, and no ambient env."""
    application = FastAPI()
    application.include_router(router_module.get_recall_coverage_router(), prefix=PREFIX)

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


def _serve_run(monkeypatch, run: RunRecord, records=()):
    """Wire ``GET /runs/{id}`` to one run, 404ing on any other id."""

    async def fake_get_run(run_id, owner_ids):
        assert OWNER_ID in owner_ids
        if run_id != run.id:
            raise CoverageRunNotFoundError()
        return run

    async def fake_questions(run_id):
        return list(records)

    monkeypatch.setattr(router_module, "get_run", fake_get_run)
    monkeypatch.setattr(router_module, "load_run_questions", fake_questions)


# --- 3: the report, and the snake_case contract -------------------------------


@pytest.fixture
def report_client(client, monkeypatch) -> TestClient:
    rows = [
        _row(text="Where are the runbooks?", coverage_score=4),
        _row(text="How do I rotate credentials?", coverage_score=2, agent_label="codex"),
        _row(text="What is our escalation path?", topic_id=None, coverage_score=0),
        _row(
            text="Which alerts page whom?",
            source=QuestionSource.USER_DEFINED.value,
            dataset_id=None,
            topic_id=None,
            coverage_score=1,
            relevance=0,
            user_id=OTHER_USER_ID,
        ),
    ]
    _serve_run(
        monkeypatch,
        _complete_run(
            rows,
            suggested=[
                SuggestedTopic(
                    suggestion_id=SUGGESTION_ID, label="Deploy rollbacks", question_count=7
                )
            ],
        ),
        [_record(row) for row in rows],
    )
    return client


def test_the_report_is_snake_case_and_never_camel_case(report_client):
    """The one contract nothing else in this repo protects. Keep these literal."""
    response = report_client.get(f"{PREFIX}/runs/{RUN_ID}")
    assert response.status_code == 200
    body = response.json()

    assert "topic_id" in body["topics"][0]
    assert "topicId" not in body["topics"][0]

    # The same rule everywhere else in the payload.
    assert "memory_score" in body
    assert "memoryScore" not in body
    assert "coverage_score" in body["questions"][0]
    assert "coverageScore" not in body["questions"][0]
    assert "first_asked_at" in body["questions"][0]
    assert "firstAskedAt" not in body["questions"][0]
    assert "recall_count" in body["run"]
    assert "recallCount" not in body["run"]
    assert "question_count" in body["suggested_topics"][0]
    assert "questionCount" not in body["suggested_topics"][0]


def test_the_report_carries_exactly_the_spec_shape(report_client):
    """Five top-level blocks, and each row is the field set the PRD names."""
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    assert set(body) == {"run", "memory_score", "topics", "suggested_topics", "questions"}
    assert set(body["run"]) == {
        "run_id",
        "status",
        "agent_label",
        "created_at",
        "finished_at",
        "question_count",
        "recall_count",
        "params",
        "error",
    }
    assert set(body["topics"][0]) == {"topic_id", "topic", "question_count", "memory_score"}
    # ``suggestion_id`` is beyond the PRD's sketch of this block, and deliberately:
    # POST /suggestions/{suggestion_id}/dismiss is keyed on it and this is the only
    # response that publishes it, so without it half the review flow is unreachable.
    assert set(body["suggested_topics"][0]) == {"suggestion_id", "label", "question_count"}
    assert set(body["questions"][0]) == {
        "question_id",
        "question",
        "coverage_score",
        "relevance",
        "topic",
        "agent",
        "user_id",
        "dataset_id",
        "dataset_name",
        "answer",
        "source",
        "first_asked_at",
        "last_asked_at",
    }


def test_the_deleted_metrics_are_absent_from_the_wire(report_client):
    """Each of these was a number the report no longer claims to know."""
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    for gone in ("datasets", "users", "sink", "benchmark_score_pct", "unscoped_ask_share"):
        assert gone not in body

    for gone in ("impact", "was_asked", "judge_answered", "question_group_id", "topic_id"):
        assert gone not in body["questions"][0]

    # retrieval_context is stored but never returned: it is up to
    # store_context_max_chars per row, and shipping it for every row would make the
    # report an order of magnitude larger than the numbers anyone came for.
    assert "retrieval_context" not in body["questions"][0]

    for gone in ("taxonomy_version", "distinct_ask_count", "collapsed_retry_count"):
        assert gone not in body["run"]


def test_no_response_model_is_an_out_dto():
    """A one-word base-class change would camel-case the whole wire contract."""
    from pydantic import BaseModel

    from cognee.api.DTO import OutDTO

    models = [
        value
        for name, value in vars(router_module).items()
        if isinstance(value, type) and issubclass(value, BaseModel)
    ]
    assert models  # the sweep found something to check
    assert [model.__name__ for model in models if issubclass(model, OutDTO)] == []


def test_answer_is_returned_only_on_the_callers_own_rows(report_client):
    """The answer is distilled from the row user's private retrieval context.

    Question text is shared across the tenant by design; the answer is not —
    returning it on a teammate's row would hand the reader dataset content
    their ACL does not grant.
    """
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    own_rows = [q for q in body["questions"] if q["user_id"] == str(OWNER_ID)]
    other_rows = [q for q in body["questions"] if q["user_id"] == str(OTHER_USER_ID)]
    assert own_rows and other_rows  # the fixture provides both

    assert all(q["answer"] == "They live in infra-docs." for q in own_rows)
    assert all(q["answer"] is None for q in other_rows)


def test_each_row_reports_its_own_agent(report_client):
    """Per row, not per run: this run covered ``claude-code`` and rows disagree.

    The default run is ``all``, and narrowing one flat table down to one agent is
    the entire point of the column — so it cannot be inherited from the run.
    """
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()
    agents = {question["question"]: question["agent"] for question in body["questions"]}

    assert agents["Where are the runbooks?"] == "claude-code"
    assert agents["How do I rotate credentials?"] == "codex"
    # A user-defined row nobody asked has no session, and therefore no agent.
    assert agents["Which alerts page whom?"] is None


def test_a_failed_run_reports_why_it_failed(client, monkeypatch):
    """Without ``run.error`` the only diagnosis path would be the server log."""
    _serve_run(monkeypatch, _failed_run())

    body = client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    assert body["run"]["status"] == "failed"
    assert body["run"]["error"] == "RuntimeError: the relational database went away"
    # The error summary is not a report: no numbers leak out of it.
    assert body["memory_score"] is None
    assert body["topics"] == []
    assert body["suggested_topics"] == []
    assert body["questions"] == []


def test_a_complete_run_reports_no_error(report_client):
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    assert body["run"]["status"] == "complete"
    assert body["run"]["error"] is None


def test_every_question_row_carries_its_source_user_and_dataset(report_client):
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    assert len(body["questions"]) == 4
    for question in body["questions"]:
        assert question["source"] in ("observed", "user_defined")
        assert question["user_id"]
        # Present on every row, and explicitly null for an unscoped one.
        assert "dataset_id" in question

    unscoped = [q for q in body["questions"] if q["dataset_id"] is None]
    assert unscoped and unscoped[0]["source"] == "user_defined"
    assert {q["user_id"] for q in body["questions"]} == {str(OWNER_ID), str(OTHER_USER_ID)}


def test_the_report_reads_the_frozen_summary_rather_than_recomputing_it(report_client):
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    # 4 and 2 on the one qualifying topic; the sink is out of the headline.
    assert body["memory_score"] == 3.0
    real = next(cell for cell in body["topics"] if cell["topic_id"] == str(TOPIC_ID))
    assert real["topic"] == "Runbooks"
    assert body["run"]["params"]["judge_score_max"] == 10
    # A per-run output, frozen alongside the topics, carrying the id the dismiss
    # route takes.
    assert body["suggested_topics"] == [
        {
            "suggestion_id": str(SUGGESTION_ID),
            "label": "Deploy rollbacks",
            "question_count": 7,
        }
    ]


def test_the_sink_is_a_topic_row_with_a_null_id(report_client):
    """It used to be a block of its own, which made every reader special-case it."""
    body = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    sink = [cell for cell in body["topics"] if cell["topic_id"] is None]
    assert len(sink) == 1
    assert sink[0]["topic"] == SINK_TOPIC_LABEL
    # Sorted last, however big it is: reading it as the biggest topic is exactly
    # the wrong impression.
    assert body["topics"][-1]["topic_id"] is None
    # The rows that landed there name it too, and there is no id literal anywhere.
    unplaceable = [q for q in body["questions"] if q["topic"] == SINK_TOPIC_LABEL]
    assert unplaceable
    assert all(q["topic"] != "other" for q in body["questions"])


def test_a_row_with_a_real_topic_is_never_labelled_uncategorized(client, monkeypatch):
    """The label map is the frozen summary, so a topic missing from it is unnameable.

    A topic whose only rows nobody asked has nothing to average, but it still
    received rows — and a response pairing a real topic with ``"Uncategorized"``
    contradicts itself: one UI groups the row under Billing, another files it under
    the sink.
    """
    speculative = uuid4()
    rows = [
        _row(text="observed", coverage_score=4),
        _row(
            text="never asked",
            source=QuestionSource.USER_DEFINED.value,
            relevance=0,
            topic_id=speculative,
            topic="Billing & invoices",
        ),
    ]
    _serve_run(monkeypatch, _complete_run(rows), [_record(row) for row in rows])

    body = client.get(f"{PREFIX}/runs/{RUN_ID}").json()
    written = next(row for row in body["questions"] if row["question"] == "never asked")

    assert written["topic"] == "Billing & invoices"
    # And the topic is in the frozen breakdown, with no asked rows to average.
    cell = next(cell for cell in body["topics"] if cell["topic_id"] == str(speculative))
    assert (cell["topic"], cell["question_count"], cell["memory_score"]) == (
        "Billing & invoices",
        0,
        None,
    )


def test_a_run_in_another_owner_scope_is_404_not_403(report_client):
    response = report_client.get(f"{PREFIX}/runs/{uuid4()}")
    assert response.status_code == 404
    assert "CoverageRunNotFoundError" in response.json()["detail"]


def test_a_summary_from_another_version_still_renders(client, monkeypatch):
    """Historical runs are the whole point of freezing the summary.

    Every field is read through ``.get``, so an unknown key is ignored and a
    missing one is null — a run persisted by a different version of ``aggregate``
    must still render rather than 500 the report.
    """
    run = _complete_run([_row()])
    run = RunRecord(
        **{
            **vars(run),
            "summary": {
                **run.summary,
                "a_field_from_the_future": 1,
                # A topic cell that has lost a key it used to carry.
                "topics": [{"topic_id": None, "question_count": 2}],
            },
        }
    )
    _serve_run(monkeypatch, run)

    response = client.get(f"{PREFIX}/runs/{RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert "a_field_from_the_future" not in body
    assert body["topics"] == [
        {"topic_id": None, "topic": SINK_TOPIC_LABEL, "question_count": 2, "memory_score": None}
    ]


def test_a_pending_run_renders_as_no_numbers_yet(client, monkeypatch):
    _serve_run(monkeypatch, _pending_run())

    body = client.get(f"{PREFIX}/runs/{RUN_ID}").json()

    assert body["run"]["status"] == "pending"
    assert body["memory_score"] is None
    assert body["topics"] == []
    assert body["suggested_topics"] == []
    assert body["questions"] == []


# --- 1: starting a run --------------------------------------------------------


def test_starting_a_run_is_202_and_pending(client, monkeypatch):
    started: list[dict] = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        started.append({"label": agent_label, "params": params, "user_id": user.id})
        return _pending_run()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    response = client.post(
        PREFIX,
        json={"agent_label": AGENT_LABEL, "params": {"max_questions": 7}},
    )

    assert response.status_code == 202
    body = response.json()
    # The receipt is enough to poll the run and nothing else: the row is pending
    # and has no counters and no score yet.
    assert set(body) == {"run_id", "status", "agent_label", "created_at"}
    assert body["status"] == "pending"
    assert body["run_id"] == str(RUN_ID)
    assert started == [{"label": AGENT_LABEL, "params": {"max_questions": 7}, "user_id": OWNER_ID}]


def test_the_trigger_route_is_the_collection_itself(app):
    """``POST /api/v1/coverage``, not ``/runs``: starting a run *is* the resource."""
    paths = app.openapi()["paths"]

    assert "post" in paths[PREFIX]
    assert "post" not in paths[f"{PREFIX}/runs"]


def test_an_omitted_agent_label_defaults_to_all(client, monkeypatch):
    seen: list = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        seen.append(agent_label)
        # What resolve_agent_scope does with None, and what the row then carries.
        return _pending_run(agent_label="all")

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    # Both an empty body and no body at all mean "all".
    assert client.post(PREFIX, json={}).json()["agent_label"] == "all"
    assert client.post(PREFIX).json()["agent_label"] == "all"
    assert seen == [None, None]


def test_a_run_already_in_flight_is_409(client, monkeypatch):
    async def fake_start(user, agent_label=None, *, params=None, config=None):
        raise CoverageRunInFlightError()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    response = client.post(PREFIX, json={"agent_label": AGENT_LABEL})

    assert response.status_code == 409
    assert "CoverageRunInFlightError" in response.json()["detail"]


def test_the_request_body_accepts_camel_case_too(client, monkeypatch):
    """Requests use InDTO, which is the one place camelCase is welcome."""
    seen: list = []

    async def fake_start(user, agent_label=None, *, params=None, config=None):
        seen.append(agent_label)
        return _pending_run()

    monkeypatch.setattr(router_module, "start_recall_coverage_run", fake_start)

    client.post(PREFIX, json={"agentLabel": AGENT_LABEL})
    assert seen == [AGENT_LABEL]


@pytest.mark.parametrize("label", ["ui", "api", "all", "claude-code", "codex"])
def test_every_reserved_and_mapped_label_resolves(client, monkeypatch, label):
    """``ui`` is a human in the cloud search box, and a first-class label.

    Exercised on ``GET /runs``, which is where the router itself resolves the label
    — so an unresolvable one would raise here rather than travel on as a string.
    ``ui`` resolves through the ordinary prefix map rather than getting a scope of
    its own: a complement-of-the-map scope is exactly ``api``, and two labels
    sharing one predicate would count, replay and judge the same traffic twice.
    """
    seen: list = []

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        seen.append(agent_label)
        return []

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    response = client.get(f"{PREFIX}/runs", params={"agent_label": label})

    assert response.status_code == 200
    assert seen == [label]


# --- 2: listing runs ----------------------------------------------------------


def test_listing_runs_applies_the_configured_default_limit(client, monkeypatch):
    calls: list[tuple] = []

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        calls.append((tuple(owner_ids), agent_label, limit))
        return [_complete_run([_row()])]

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    body = client.get(f"{PREFIX}/runs").json()
    assert len(body) == 1
    assert body[0]["agent_label"] == AGENT_LABEL
    assert calls[0][1] is None
    assert calls[0][2] == _config().runs_list_default_limit

    client.get(f"{PREFIX}/runs", params={"agent_label": AGENT_LABEL, "limit": 2})
    assert calls[1][1] == AGENT_LABEL
    assert calls[1][2] == 2


def test_each_history_item_carries_the_headline_score(client, monkeypatch):
    """What makes the list a trend rather than a log — read off the frozen summary."""

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        return [_complete_run([_row(coverage_score=6)]), _pending_run(), _failed_run()]

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    body = client.get(f"{PREFIX}/runs").json()

    assert set(body[0]) == {
        "run_id",
        "status",
        "agent_label",
        "created_at",
        "finished_at",
        "question_count",
        "memory_score",
    }
    assert body[0]["memory_score"] == 6.0
    assert body[0]["question_count"] == 1
    # A run that has not finished has no score, and that is null rather than 0.0.
    assert body[1]["status"] == "pending"
    assert body[1]["memory_score"] is None
    # A failed one is listed, with no score and without its reason: ``error`` is a
    # detail-route field, and the history list is the seven fields above.
    assert body[2]["status"] == "failed"
    assert body[2]["memory_score"] is None
    assert "error" not in body[2]


def test_an_unknown_agent_label_is_422_everywhere_it_is_accepted(client, monkeypatch):
    """A typo must not be indistinguishable from 'this agent asked nothing'.

    422 rather than 404 because the label is a parameter *value*: the route exists,
    and the value is not one it accepts.
    """

    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        raise AssertionError("an unknown label must never reach the repository")

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    response = client.get(f"{PREFIX}/runs", params={"agent_label": "claude-codex"})

    assert response.status_code == 422
    assert "UnknownAgentLabelError" in response.json()["detail"]


def test_a_valid_label_with_no_runs_is_an_empty_list_not_an_error(client, monkeypatch):
    async def fake_list(owner_ids, agent_label=None, *, limit=None):
        return []

    monkeypatch.setattr(router_module, "list_runs", fake_list)

    response = client.get(f"{PREFIX}/runs", params={"agent_label": "codex"})
    assert response.status_code == 200
    assert response.json() == []


def test_a_non_positive_limit_is_422_rather_than_a_negative_sql_limit(client):
    """``LIMIT -1`` is "no limit" on SQLite and an error on Postgres."""
    assert client.get(f"{PREFIX}/runs", params={"limit": -1}).status_code == 422
    assert client.get(f"{PREFIX}/runs", params={"limit": 0}).status_code == 422


# --- 7, 8, 9: topics ----------------------------------------------------------


def test_creating_a_topic_is_201_and_returns_the_minted_id(client, monkeypatch):
    posted: list[tuple] = []

    async def fake_create(owner_id, label, *, config=None):
        posted.append((owner_id, label))
        return _topic(label=label), None

    monkeypatch.setattr(router_module, "create_topic_from_label", fake_create)

    response = client.post(f"{PREFIX}/topics", json={"topic": "Deploy rollbacks"})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"topic_id", "topic", "created_at"}
    assert body["topic_id"] == str(TOPIC_ID)
    assert body["topic"] == "Deploy rollbacks"
    assert posted == [(OWNER_ID, "Deploy rollbacks")]


def test_posting_a_suggested_label_is_the_accept_path(client, monkeypatch):
    """One route for both of the UI's flows: they type a name, or click a proposed one.

    From the owner's side it is one act, and there is no separate accept route — a
    second way to mint a topic id would be a second chance to mint two for one
    theme.
    """
    accepted = _suggestion()

    async def fake_create(owner_id, label, *, config=None):
        return _topic(label="Credential rotation"), accepted

    monkeypatch.setattr(router_module, "create_topic_from_label", fake_create)

    response = client.post(f"{PREFIX}/topics", json={"topic": "Credential rotation"})

    assert response.status_code == 201
    # The response is the same shape either way: the caller asked for a topic and
    # got one, and which path it took is not their business.
    assert response.json() == {
        "topic_id": str(TOPIC_ID),
        "topic": "Credential rotation",
        "created_at": BASE_TIME.isoformat().replace("+00:00", "Z"),
    }


def test_there_is_no_accept_route_any_more(app):
    """``POST /topics`` replaced it, so the second minting path is gone."""
    paths = app.openapi()["paths"]

    assert f"{PREFIX}/suggestions/{{suggestion_id}}/accept" not in paths


def test_a_blank_topic_is_422_and_a_duplicate_is_409(client, monkeypatch):
    """A duplicate label does not add a topic, it silently disables one.

    Two topics with near-identical centroids cannot be separated by the assignment
    margin rule, so every question about that theme lands in ``Uncategorized``.
    """

    async def blank(owner_id, label, *, config=None):
        raise EmptyTopicLabelError()

    monkeypatch.setattr(router_module, "create_topic_from_label", blank)
    response = client.post(f"{PREFIX}/topics", json={"topic": "   "})
    assert response.status_code == 422
    assert "EmptyTopicLabelError" in response.json()["detail"]

    async def duplicate(owner_id, label, *, config=None):
        raise DuplicateTopicError()

    monkeypatch.setattr(router_module, "create_topic_from_label", duplicate)
    response = client.post(f"{PREFIX}/topics", json={"topic": "Runbooks"})
    assert response.status_code == 409
    assert "DuplicateTopicError" in response.json()["detail"]


def test_listing_topics_is_owner_scoped_and_carries_a_question_count(client, monkeypatch):
    other = uuid4()

    async def fake_topics(owner_ids, *, include_deleted=False):
        assert OWNER_ID in owner_ids
        # Soft-deleted topics are never listed: the row survives only so a
        # historical run can still resolve the topic id on its own rows.
        assert include_deleted is False
        return [_topic(), _topic(other, "Deploy rollbacks")]

    async def fake_counts(owner_ids):
        return {TOPIC_ID: 4}

    monkeypatch.setattr(router_module, "list_topics", fake_topics)
    monkeypatch.setattr(router_module, "topic_question_counts", fake_counts)

    body = client.get(f"{PREFIX}/topics").json()

    assert [item["topic_id"] for item in body] == [str(TOPIC_ID), str(other)]
    assert set(body[0]) == {"topic_id", "topic", "question_count", "created_at"}
    assert body[0]["question_count"] == 4
    # A topic nothing has landed in — a freshly created one above all — honestly
    # reports 0 rather than being hidden.
    assert body[1]["question_count"] == 0


def test_listing_topics_no_longer_returns_the_suggestions(app, client, monkeypatch):
    """They are a per-run output and travel in the run report instead."""

    async def fake_topics(owner_ids, *, include_deleted=False):
        return [_topic()]

    async def fake_counts(owner_ids):
        return {}

    monkeypatch.setattr(router_module, "list_topics", fake_topics)
    monkeypatch.setattr(router_module, "topic_question_counts", fake_counts)

    body = client.get(f"{PREFIX}/topics").json()

    # A bare list, not an object with a suggestions half or a version counter.
    assert isinstance(body, list)
    assert "suggestions" not in body[0]
    assert "taxonomy_version" not in body[0]
    # And no query parameters at all — no agent_label, because one taxonomy serves
    # every one of an owner's agents, which is what makes two agents' per-topic
    # scores comparable; and no include_deleted, because a soft-deleted row exists
    # only so history resolves its ids and the item shape has nowhere to show it.
    assert app.openapi()["paths"][f"{PREFIX}/topics"]["get"].get("parameters", []) == []


def test_deleting_a_topic_is_204_with_no_body(client, monkeypatch):
    """Nothing to report back: there is no version counter any more.

    The topic's questions are never deleted — they fall back to ``Uncategorized``
    on the next run, which is exactly the signal that row exists to give.
    """
    deleted: list = []

    async def fake_delete(topic_id, owner_ids):
        assert OWNER_ID in owner_ids
        deleted.append(topic_id)

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    response = client.delete(f"{PREFIX}/topics/{TOPIC_ID}")

    assert response.status_code == 204
    assert not response.content
    assert deleted == [TOPIC_ID]


def test_deleting_a_topic_in_another_owner_scope_is_404(client, monkeypatch):
    async def fake_delete(topic_id, owner_ids):
        raise CoverageTopicNotFoundError()

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    response = client.delete(f"{PREFIX}/topics/{uuid4()}")
    assert response.status_code == 404
    assert "CoverageTopicNotFoundError" in response.json()["detail"]


def test_an_unparseable_topic_id_is_404_not_422(client, monkeypatch):
    """It names nothing, and telling the caller how ids look helps them not at all.

    The sink is in the same position by construction: it is reported with
    ``topic_id: null``, so there is no id to address it by and nothing to
    special-case here.
    """

    async def fake_delete(topic_id, owner_ids):
        raise AssertionError("an unparseable id must never reach the repository")

    monkeypatch.setattr(router_module, "delete_topic", fake_delete)

    for not_an_id in ("not-a-uuid", "other", "Uncategorized"):
        assert client.delete(f"{PREFIX}/topics/{not_an_id}").status_code == 404


# --- 10: dismissing a suggestion ---------------------------------------------


def test_dismissing_a_suggestion_is_204_with_no_body(client, monkeypatch):
    """The row is kept because it *is* the decision — the guard reads it next run."""
    dismissed: list = []

    async def fake_dismiss(suggestion_id, owner_ids):
        assert OWNER_ID in owner_ids
        dismissed.append(suggestion_id)
        return _suggestion(status=SuggestionStatus.DISMISSED.value)

    monkeypatch.setattr(router_module, "dismiss_topic_suggestion", fake_dismiss)

    suggestion_id = uuid4()
    response = client.post(f"{PREFIX}/suggestions/{suggestion_id}/dismiss")

    assert response.status_code == 204
    assert not response.content
    assert dismissed == [suggestion_id]


def test_dismissing_a_decided_suggestion_is_409_and_a_missing_one_404(client, monkeypatch):
    async def not_pending(suggestion_id, owner_ids):
        raise CoverageSuggestionNotPendingError()

    monkeypatch.setattr(router_module, "dismiss_topic_suggestion", not_pending)
    assert client.post(f"{PREFIX}/suggestions/{uuid4()}/dismiss").status_code == 409

    async def missing(suggestion_id, owner_ids):
        raise CoverageSuggestionNotFoundError()

    monkeypatch.setattr(router_module, "dismiss_topic_suggestion", missing)
    assert client.post(f"{PREFIX}/suggestions/{uuid4()}/dismiss").status_code == 404


def test_the_id_the_report_publishes_is_the_id_dismiss_takes(report_client, monkeypatch):
    """Reachability. The run report is the only response carrying a suggestion id.

    A client can therefore only dismiss what a report named, so the two must be the
    same value. Asserted end to end because they live in different models and
    nothing else connects them: drop ``suggestion_id`` from the report and this
    route silently addresses nothing.
    """
    dismissed: list = []

    async def fake_dismiss(suggestion_id, owner_ids):
        dismissed.append(suggestion_id)
        return _suggestion(status=SuggestionStatus.DISMISSED.value)

    monkeypatch.setattr(router_module, "dismiss_topic_suggestion", fake_dismiss)

    report = report_client.get(f"{PREFIX}/runs/{RUN_ID}").json()
    published = report["suggested_topics"][0]["suggestion_id"]

    assert report_client.post(f"{PREFIX}/suggestions/{published}/dismiss").status_code == 204
    assert dismissed == [SUGGESTION_ID]


# --- 4, 5, 6: user-defined questions -----------------------------------------


def test_adding_a_question_is_201(client, monkeypatch):
    seen: list = []

    async def fake_create(user, question, *, config=None):
        seen.append((user.id, question))
        return _curated()

    monkeypatch.setattr(router_module, "create_curated_question", fake_create)

    response = client.post(
        f"{PREFIX}/questions",
        json={"question": "What is our escalation path out of hours?"},
    )

    assert response.status_code == 201
    body = response.json()
    # One flat list per owner: no scope, and no agent label.
    assert set(body) == {"question_id", "question", "created_at"}
    assert body["question"].startswith("What is our escalation")
    assert seen == [(OWNER_ID, "What is our escalation path out of hours?")]


def test_a_duplicate_question_is_409(client, monkeypatch):
    """Refused rather than merged, so the writer learns it is already covered."""

    async def fake_create(user, question, *, config=None):
        raise DuplicateCuratedQuestionError()

    monkeypatch.setattr(router_module, "create_curated_question", fake_create)

    response = client.post(f"{PREFIX}/questions", json={"question": "Where are the runbooks?"})

    assert response.status_code == 409
    assert "DuplicateCuratedQuestionError" in response.json()["detail"]


def test_a_full_question_list_is_422_not_409(client, monkeypatch):
    """Pinned next to the duplicate so the two cannot drift apart unnoticed.

    A duplicate is a conflict with an existing row; a full list makes the posted
    value unprocessable. The route's docstring names both codes, and nothing else
    checks that the exception still agrees with it.
    """

    async def fake_create(user, question, *, config=None):
        raise CuratedQuestionLimitError()

    monkeypatch.setattr(router_module, "create_curated_question", fake_create)

    response = client.post(f"{PREFIX}/questions", json={"question": "One question too many."})

    assert response.status_code == 422
    assert "CuratedQuestionLimitError" in response.json()["detail"]


def test_listing_questions_is_the_callers_own_flat_list(client, monkeypatch):
    async def fake_list(user):
        assert user.id == OWNER_ID
        return [_curated(), _curated()]

    monkeypatch.setattr(router_module, "list_curated_questions", fake_list)

    response = client.get(f"{PREFIX}/questions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(set(item) == {"question_id", "question", "created_at"} for item in body)


def test_deleting_a_question_is_204_and_404_out_of_scope(client, monkeypatch):
    deleted: list = []

    async def fake_delete(user, question_id):
        deleted.append(question_id)

    monkeypatch.setattr(router_module, "delete_curated_question", fake_delete)

    question_id = uuid4()
    response = client.delete(f"{PREFIX}/questions/{question_id}")
    assert response.status_code == 204
    assert not response.content
    assert deleted == [question_id]

    async def fake_missing(user, question_id):
        raise CuratedQuestionNotFoundError()

    monkeypatch.setattr(router_module, "delete_curated_question", fake_missing)
    assert client.delete(f"{PREFIX}/questions/{uuid4()}").status_code == 404


# --- OpenAPI ------------------------------------------------------------------


def test_every_route_of_the_spec_is_mounted(app):
    """Exactly ten operations, at ``/api/v1/coverage``, and nothing else."""
    schema = app.openapi()
    operations = {
        (method.upper(), path.replace(PREFIX, "") or "")
        for path, methods in schema["paths"].items()
        for method in methods
    }

    assert operations == {
        ("POST", ""),
        ("GET", "/runs"),
        ("GET", "/runs/{run_id}"),
        ("POST", "/questions"),
        ("GET", "/questions"),
        ("DELETE", "/questions/{question_id}"),
        ("POST", "/topics"),
        ("GET", "/topics"),
        ("DELETE", "/topics/{topic_id}"),
        ("POST", "/suggestions/{suggestion_id}/dismiss"),
    }


def test_the_deleted_routes_are_gone(app):
    """``/agents`` and the benchmark matrix; the old names for the survivors."""
    paths = set(app.openapi()["paths"])

    for gone in ("/agents", "/summary", "/curated-questions"):
        assert f"{PREFIX}{gone}" not in paths


def test_the_declared_schema_is_snake_case(app):
    """A second, request-free guard: the OpenAPI components must not be camel.

    Only response models: the request bodies are ``InDTO``s and legitimately
    publish camelCase aliases, which is a convenience on the way in and says
    nothing about what goes out.
    """
    components = app.openapi()["components"]["schemas"]

    for name in (
        "QuestionRow",
        "TopicScoreItem",
        "SuggestedTopicItem",
        "RunInfo",
        "RunListItem",
        "StartedRun",
        "CoverageReport",
        "CreatedTopic",
        "TopicItem",
        "UserQuestionItem",
    ):
        properties = components[name]["properties"]
        assert not [key for key in properties if key != key.lower()], (name, properties.keys())
