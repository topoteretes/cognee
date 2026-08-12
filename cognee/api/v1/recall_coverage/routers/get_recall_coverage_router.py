"""HTTP router for recall coverage — the twelve routes of spec section 5.

Given an agent (a tool such as Claude Code or Codex), replay the questions it
recently asked, judge whether memory could answer them, and report where the gaps
are. Runs are always background: ``POST /runs`` inserts a ``pending`` row,
schedules the coroutine and returns **202**; ``GET /runs/{run_id}`` is the main
read.

Four rules hold across every route here:

* **Response models are plain ``pydantic.BaseModel``, never ``OutDTO``.** ``OutDTO``
  sets ``alias_generator=to_camel`` and FastAPI serializes response models *by
  alias*, so an ``OutDTO`` here would emit ``topicId`` / ``judgeScore`` and break
  the agreed snake_case wire contract (see ``cognee/api/DTO.py``). Request bodies
  do use ``InDTO``, which accepts both cases — that is a convenience on the way in,
  and it does not touch what goes out.
* **The report is read from the run's frozen ``summary``**, never recomputed.
  Recomputing would let a deleted topic, or an owner losing access to a dataset,
  silently reshape a historical run and destroy the trend that stable topic ids
  exist to carry.
* **Authorisation is owner scope, not ``Depends(get_authenticated_user)`` alone.**
  That dependency falls back to the default user when authentication is optional,
  so every id-keyed route filters on **both** id and owner scope and 404s on a
  mismatch — never 403, which would confirm that another owner's row with that id
  exists. Errors are raised as ``CogneeApiError`` subclasses and turned into
  responses by the global handler in ``cognee/api/client.py``.
* **An unknown ``agent_label`` is a 404; a valid label with no traffic is an empty
  run, not an error.** A typo must not be indistinguishable from "this agent has
  asked nothing yet", which is a legitimate answer — and, until
  ``Query.session_id`` ships, the expected one for every label except ``all``.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.agents import agent_window_counts
from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.pipeline import build_params, start_recall_coverage_run
from cognee.modules.recall_coverage.repository import (
    BenchmarkCell,
    CuratedQuestion,
    QuestionRecord,
    RunRecord,
    SuggestionRecord,
    TopicRecord,
    accept_topic_suggestion,
    benchmark_cells,
    create_curated_question,
    curated_owner_ids,
    current_taxonomy_version,
    delete_curated_question,
    delete_topic,
    dismiss_topic_suggestion,
    get_run,
    latest_complete_runs,
    list_curated_questions,
    list_runs,
    list_topics,
    load_pending_suggestions,
    load_run_questions,
    parse_topic_id,
)
from cognee.modules.recall_coverage.types import SINK_TOPIC_ID, SINK_TOPIC_LABEL
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger("recall_coverage")


# --- response models (plain BaseModel: the wire contract is snake_case) -------


class ErrorResponse(BaseModel):
    """Generic API error response, as produced by the global exception handler."""

    detail: str


class RunInfo(BaseModel):
    """One ``recall_coverage_runs`` row, counters included."""

    run_id: str
    agent_label: str
    status: str
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    recall_row_count: int = 0
    distinct_ask_count: int = 0
    collapsed_retry_count: int = 0
    question_row_count: int = 0
    curated_question_count: int = 0
    topic_count: int = 0
    dataset_count: int = 0
    user_count: int = 0
    taxonomy_version: int = 0
    params: Optional[dict] = None


class AlertItem(BaseModel):
    """One thing worth telling the reader, as a stable code plus prose."""

    code: str
    message: str


class TopicCell(BaseModel):
    """One topic's aggregate. ``avg_score`` is null below the scored-row minimum."""

    topic_id: str
    label: str
    question_count: int = 0
    scored_question_count: int = 0
    avg_score: Optional[float] = None


class SinkCell(BaseModel):
    """The residual: questions the taxonomy could not place."""

    topic_id: str = SINK_TOPIC_ID
    label: str = SINK_TOPIC_LABEL
    question_count: int = 0
    scored_question_count: int = 0
    share: Optional[float] = None
    avg_score: Optional[float] = None
    alerts: list[AlertItem] = Field(default_factory=list)


class DatasetCell(BaseModel):
    """One dataset's aggregate. ``dataset_id`` is null for the unscoped rows."""

    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    question_count: int = 0
    scored_question_count: int = 0
    avg_score: Optional[float] = None


class UserCell(BaseModel):
    """One asker's aggregate. A run covers every user, and the UI filters."""

    user_id: str
    question_count: int = 0
    scored_question_count: int = 0
    avg_score: Optional[float] = None


class QuestionRow(BaseModel):
    """One judged question row: ``(user_id, dataset_id, canonical text)``.

    Raw rows, deliberately ungrouped — grouping and filtering are the UI's job,
    which is what ``question_group_id`` is for. ``retrieval_context`` is stored but
    not returned: it is up to ``store_context_max_chars`` per row, and shipping it
    for every row would make the report an order of magnitude larger than the
    numbers anyone came for.
    """

    question_id: str
    question_group_id: Optional[str] = None
    source: str
    was_asked: bool
    question_text: str
    user_id: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    answer: Optional[str] = None
    judge_score: Optional[int] = None
    judge_answered: Optional[bool] = None
    # Why the scores are null on this row, when they are.
    error: Optional[str] = None
    topic_id: str = SINK_TOPIC_ID
    topic_label: str = SINK_TOPIC_LABEL
    first_asked_at: Optional[datetime] = None
    last_asked_at: Optional[datetime] = None
    occurrence_count: int = 0
    impact: Optional[float] = None


class CoverageReport(BaseModel):
    """The full report for one run: the frozen summary plus its raw rows."""

    run: RunInfo
    overall_score: Optional[float] = None
    benchmark_score_pct: Optional[float] = None
    unscoped_ask_share: Optional[float] = None
    sink: SinkCell = Field(default_factory=SinkCell)
    datasets: list[DatasetCell] = Field(default_factory=list)
    users: list[UserCell] = Field(default_factory=list)
    topics: list[TopicCell] = Field(default_factory=list)
    questions: list[QuestionRow] = Field(default_factory=list)


class AgentSummary(BaseModel):
    """One label with traffic in the window, plus its latest complete run."""

    agent_label: str
    recall_row_count: int = 0
    latest_run: Optional[RunInfo] = None
    overall_score: Optional[float] = None


class TopicItem(BaseModel):
    """One stored topic. ``deleted_at`` is set on a soft-deleted one."""

    topic_id: str
    label: str
    seed_question_count: int = 0
    taxonomy_version: int = 0
    created_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None


class SuggestionItem(BaseModel):
    """One topic suggestion. ``cohesion`` orders candidates and is never scored."""

    suggestion_id: str
    label: str
    status: str
    question_count: int = 0
    cohesion: Optional[float] = None
    agent_label: Optional[str] = None
    run_id: Optional[str] = None
    accepted_topic_id: Optional[str] = None
    created_at: Optional[datetime] = None


class TopicsResponse(BaseModel):
    """The owner's taxonomy plus its undecided suggestions.

    The pending suggestions travel with the topics rather than on a route of their
    own: they are the only place the ids that ``POST /suggestions/{id}/accept``
    takes come from, and a review screen shows both halves at once.
    """

    taxonomy_version: int = 0
    topics: list[TopicItem] = Field(default_factory=list)
    suggestions: list[SuggestionItem] = Field(default_factory=list)


class TaxonomyVersionResponse(BaseModel):
    """What a delete returns: the owner's new, monotone taxonomy version."""

    taxonomy_version: int


class CuratedQuestionItem(BaseModel):
    """One human-added question. ``agent_label`` is null for a shared one."""

    question_id: str
    scope: str
    agent_label: Optional[str] = None
    question_text: str
    created_at: Optional[datetime] = None


class BenchmarkMatrixCell(BaseModel):
    """One datasets x agents cell over shared curated rows only."""

    agent_label: str
    run_id: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    question_count: int = 0
    scored_question_count: int = 0
    avg_score: Optional[float] = None
    score_pct: Optional[float] = None


class BenchmarkMatrix(BaseModel):
    """The benchmark matrix: which agent answers which dataset's questions."""

    judge_score_max: int
    agent_labels: list[str] = Field(default_factory=list)
    cells: list[BenchmarkMatrixCell] = Field(default_factory=list)


# --- request models (InDTO: accepts snake_case and camelCase) ----------------


class StartRunRequest(InDTO):
    """``agent_label`` omitted defaults to ``all`` — every recall in the window."""

    agent_label: Optional[str] = Field(
        default=None,
        description=(
            "Agent to analyse: a label from the configured prefix map, or the "
            "literal 'api' / 'all'. Omitted means 'all'."
        ),
    )
    params: Optional[dict] = Field(
        default=None,
        description="Per-run overrides of the deployment's coverage parameters.",
    )


class CreateCuratedQuestionRequest(InDTO):
    """A question a human says memory should answer."""

    question_text: str = Field(description="The question, as a human would ask it.")
    scope: str = Field(
        default="agent",
        description=(
            "'agent' (one agent_label, which is then required) or 'shared' (the "
            "benchmark set, which forbids one)."
        ),
    )
    agent_label: Optional[str] = Field(
        default=None, description="Required when scope is 'agent', forbidden when it is 'shared'."
    )


# --- mapping ------------------------------------------------------------------


def _owner_scope(user: Any) -> tuple[UUID, ...]:
    """Owner ids this caller may read: their own, plus their tenant's."""
    return curated_owner_ids(user)


def _run_info(run: RunRecord) -> RunInfo:
    return RunInfo(
        run_id=str(run.id),
        agent_label=run.agent_label,
        status=run.status,
        created_at=run.created_at,
        finished_at=run.finished_at,
        recall_row_count=run.recall_row_count,
        distinct_ask_count=run.distinct_ask_count,
        collapsed_retry_count=run.collapsed_retry_count,
        question_row_count=run.question_row_count,
        curated_question_count=run.curated_question_count,
        topic_count=run.topic_count,
        dataset_count=run.dataset_count,
        user_count=run.user_count,
        taxonomy_version=run.taxonomy_version,
        params=run.params,
    )


def _summary_of(run: RunRecord) -> dict:
    """The run's frozen summary, or an empty mapping.

    A pending run has no summary and a failed one carries ``{"error": ...}``, so
    every read below goes through ``.get`` — a report for an unfinished run must
    render as "no numbers yet" rather than raise.
    """
    summary = run.summary if isinstance(run.summary, dict) else {}
    return summary if "topics" in summary or "sink" in summary else {}


def _topic_labels(summary: Mapping[str, Any]) -> dict[str, str]:
    """Topic id -> label, straight out of the frozen summary.

    The labels come from the summary rather than from a join on
    ``recall_coverage_topics`` precisely because a topic may since have been
    deleted, and the report must still name what it measured.
    """
    labels = {
        str(topic.get("topic_id")): str(topic.get("label", ""))
        for topic in summary.get("topics", [])
        if topic.get("topic_id")
    }
    labels[SINK_TOPIC_ID] = SINK_TOPIC_LABEL
    return labels


def _question_row(record: QuestionRecord, labels: Mapping[str, str]) -> QuestionRow:
    topic_id = SINK_TOPIC_ID if record.topic_id is None else str(record.topic_id)
    return QuestionRow(
        question_id=str(record.id),
        question_group_id=None
        if record.question_group_id is None
        else str(record.question_group_id),
        source=record.source,
        was_asked=record.was_asked,
        question_text=record.question_text,
        user_id=str(record.user_id),
        dataset_id=None if record.dataset_id is None else str(record.dataset_id),
        dataset_name=record.dataset_name,
        answer=record.answer,
        judge_score=record.judge_score,
        judge_answered=record.judge_answered,
        error=record.error,
        topic_id=topic_id,
        topic_label=labels.get(topic_id, SINK_TOPIC_LABEL),
        first_asked_at=record.first_asked_at,
        last_asked_at=record.last_asked_at,
        occurrence_count=record.occurrence_count,
        impact=record.impact,
    )


def _report(run: RunRecord, questions: Sequence[QuestionRecord]) -> CoverageReport:
    summary = _summary_of(run)
    labels = _topic_labels(summary)

    return CoverageReport(
        run=_run_info(run),
        overall_score=summary.get("overall_score"),
        benchmark_score_pct=summary.get("benchmark_score_pct"),
        unscoped_ask_share=summary.get("unscoped_ask_share"),
        sink=SinkCell(**summary["sink"]) if summary.get("sink") else SinkCell(),
        datasets=[DatasetCell(**cell) for cell in summary.get("datasets", [])],
        users=[UserCell(**cell) for cell in summary.get("users", [])],
        topics=[TopicCell(**cell) for cell in summary.get("topics", [])],
        questions=[_question_row(record, labels) for record in questions],
    )


def _topic_item(topic: TopicRecord) -> TopicItem:
    return TopicItem(
        topic_id=str(topic.id),
        label=topic.label,
        seed_question_count=topic.seed_question_count,
        taxonomy_version=topic.taxonomy_version,
        created_at=topic.created_at,
        deleted_at=topic.deleted_at,
    )


def _suggestion_item(suggestion: SuggestionRecord) -> SuggestionItem:
    return SuggestionItem(
        suggestion_id=str(suggestion.id),
        label=suggestion.label,
        status=suggestion.status,
        question_count=suggestion.question_count,
        cohesion=suggestion.cohesion,
        agent_label=suggestion.agent_label,
        run_id=None if suggestion.run_id is None else str(suggestion.run_id),
        accepted_topic_id=None
        if suggestion.accepted_topic_id is None
        else str(suggestion.accepted_topic_id),
        created_at=suggestion.created_at,
    )


def _curated_item(question: CuratedQuestion) -> CuratedQuestionItem:
    return CuratedQuestionItem(
        question_id=str(question.id),
        scope=question.scope,
        agent_label=question.agent_label,
        question_text=question.question_text,
        created_at=question.created_at,
    )


def _matrix_cell(cell: BenchmarkCell, *, judge_score_max: int) -> BenchmarkMatrixCell:
    """One cell, with the percentage of ``judge_score_max`` alongside the mean."""
    score_pct = (
        None
        if cell.avg_score is None or judge_score_max <= 0
        else (cell.avg_score / judge_score_max) * 100.0
    )
    return BenchmarkMatrixCell(
        agent_label=cell.agent_label,
        run_id=str(cell.run_id),
        dataset_id=None if cell.dataset_id is None else str(cell.dataset_id),
        dataset_name=cell.dataset_name,
        question_count=cell.question_count,
        scored_question_count=cell.scored_question_count,
        avg_score=cell.avg_score,
        score_pct=score_pct,
    )


def _requested_labels(agent_labels: Optional[str], user: Any, config: Any) -> list[str]:
    """Parse and validate ``?agent_labels=a,b,c``. An unknown label 404s."""
    if not agent_labels or not agent_labels.strip():
        return []
    return [
        resolve_agent_scope(part.strip(), user=user, config=config).label
        for part in agent_labels.split(",")
        if part.strip()
    ]


def _telemetry(event: str, user: Any, **properties: Any) -> None:
    send_telemetry(
        event,
        getattr(user, "id", None),
        additional_properties={"cognee_version": cognee_version, **properties},
    )


# --- the router ---------------------------------------------------------------

_ERRORS = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


def get_recall_coverage_router() -> APIRouter:
    router = APIRouter()

    def _config() -> RecallCoverageConfig:
        return get_recall_coverage_config()

    # 1 --------------------------------------------------------------------
    @router.post("/runs", response_model=RunInfo, status_code=202, responses=_ERRORS)
    async def start_run(
        payload: Optional[StartRunRequest] = None,
        user: User = Depends(get_authenticated_user),
    ) -> RunInfo:
        """Start a coverage run in the background and return its ``pending`` row.

        Always background — a run replays and judges every question row, which is
        minutes of LLM calls. **409** when a run for this ``(owner, agent_label)``
        is already pending or running: overlapping runs would multiply the cost and
        race on the same taxonomy.
        """
        request = payload or StartRunRequest()

        run = await start_recall_coverage_run(
            user,
            request.agent_label,
            params=request.params,
            config=_config(),
        )

        _telemetry(
            "Recall Coverage Run Started API Endpoint Invoked",
            user,
            endpoint="POST /v1/recall-coverage/runs",
            agent_label=run.agent_label,
            run_id=str(run.id),
        )
        return _run_info(run)

    # 2 --------------------------------------------------------------------
    @router.get("/runs", response_model=list[RunInfo], responses=_ERRORS)
    async def list_coverage_runs(
        agent_label: Optional[str] = Query(
            default=None, description="Narrow the list to one agent label."
        ),
        limit: Optional[int] = Query(
            default=None, description="Newest N runs. Defaults to runs_list_default_limit."
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[RunInfo]:
        """The caller's runs, newest first."""
        config = _config()
        label = (
            resolve_agent_scope(agent_label, user=user, config=config).label
            if agent_label
            else None
        )

        runs = await list_runs(
            _owner_scope(user),
            label,
            limit=limit if limit is not None else config.runs_list_default_limit,
        )
        return [_run_info(run) for run in runs]

    # 3 --------------------------------------------------------------------
    @router.get("/runs/{run_id}", response_model=CoverageReport, responses=_ERRORS)
    async def get_coverage_report(
        run_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> CoverageReport:
        """The full report: the run's frozen summary plus its raw question rows.

        The main read endpoint. Rows come back ungrouped, newest demand first,
        curated rows pinned above observed ones; grouping and filtering are the
        UI's job and ``question_group_id`` is what it groups on.
        """
        run = await get_run(run_id, _owner_scope(user))
        questions = await load_run_questions(run.id)

        _telemetry(
            "Recall Coverage Report Read API Endpoint Invoked",
            user,
            endpoint="GET /v1/recall-coverage/runs/{run_id}",
            agent_label=run.agent_label,
            run_id=str(run.id),
        )
        return _report(run, questions)

    # 4 --------------------------------------------------------------------
    @router.get("/agents", response_model=list[AgentSummary], responses=_ERRORS)
    async def list_agents(
        limit: Optional[int] = Query(
            default=None,
            description="Top N by window traffic. Defaults to agents_list_default_limit.",
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[AgentSummary]:
        """Labels that actually asked something, busiest first.

        Discovered from traffic, never from a registry: an agent exists because it
        asked something. Each row is left-joined to that label's latest **complete**
        run, so a label with traffic but no finished run still appears — with a null
        run rather than being hidden.
        """
        config = _config()
        params = build_params(None, config)
        since = datetime.now(timezone.utc) - timedelta(days=params.max_age_days)

        windows = await agent_window_counts(
            since=since, query_types=params.query_types, config=config
        )
        top = windows[: limit if limit is not None else config.agents_list_default_limit]

        # Nothing to join when no label has traffic — and an empty label list would
        # make ``latest_complete_runs`` fall back to "every label", which is a full
        # scan to answer a question nobody asked.
        latest = (
            await latest_complete_runs(_owner_scope(user), [window.label for window in top])
            if top
            else {}
        )

        summaries: list[AgentSummary] = []
        for window in top:
            run = latest.get(window.label)
            summaries.append(
                AgentSummary(
                    agent_label=window.label,
                    recall_row_count=window.recall_row_count,
                    latest_run=None if run is None else _run_info(run),
                    overall_score=None if run is None else _summary_of(run).get("overall_score"),
                )
            )
        return summaries

    # 5 --------------------------------------------------------------------
    @router.get("/topics", response_model=TopicsResponse, responses=_ERRORS)
    async def get_topics(
        include_deleted: bool = Query(default=False, description="Include soft-deleted topics."),
        user: User = Depends(get_authenticated_user),
    ) -> TopicsResponse:
        """The owner's taxonomy plus its pending suggestions.

        Owner-scoped and with **no ``agent_label`` parameter**: one taxonomy serves
        all of an owner's agents, which is what makes "Codex 4.2 on Billing, Claude
        Code 2.1 on Billing" a sentence at all.
        """
        owners = _owner_scope(user)
        topics = await list_topics(owners, include_deleted=include_deleted)
        suggestions = await load_pending_suggestions(user.id)

        return TopicsResponse(
            taxonomy_version=await current_taxonomy_version(user.id),
            topics=[_topic_item(topic) for topic in topics],
            suggestions=[_suggestion_item(suggestion) for suggestion in suggestions],
        )

    # 6 --------------------------------------------------------------------
    @router.delete("/topics/{topic_id}", response_model=TaxonomyVersionResponse, responses=_ERRORS)
    async def remove_topic(
        topic_id: str,
        user: User = Depends(get_authenticated_user),
    ) -> TaxonomyVersionResponse:
        """Soft-delete a topic and return the owner's new taxonomy version.

        **422** for the sink: ``"other"`` is a wire literal, not a row. The topic's
        questions are never deleted — they fall back to the sink on the next run,
        which is exactly the signal the sink exists to give.
        """
        parsed = parse_topic_id(topic_id)
        version = await delete_topic(parsed, _owner_scope(user))

        _telemetry(
            "Recall Coverage Topic Deleted API Endpoint Invoked",
            user,
            endpoint="DELETE /v1/recall-coverage/topics/{topic_id}",
            topic_id=str(parsed),
        )
        return TaxonomyVersionResponse(taxonomy_version=version)

    # 7 --------------------------------------------------------------------
    @router.post("/suggestions/{suggestion_id}/accept", response_model=TopicItem, responses=_ERRORS)
    async def accept_suggestion(
        suggestion_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> TopicItem:
        """Accept a suggestion. **This call mints the topic id** and bumps the version.

        **409** on an already decided suggestion: a second accept would mint a
        second id for one theme and split its score trend in half.
        """
        topic, _suggestion = await accept_topic_suggestion(suggestion_id, _owner_scope(user))

        _telemetry(
            "Recall Coverage Suggestion Accepted API Endpoint Invoked",
            user,
            endpoint="POST /v1/recall-coverage/suggestions/{id}/accept",
            suggestion_id=str(suggestion_id),
            topic_id=str(topic.id),
        )
        return _topic_item(topic)

    # 8 --------------------------------------------------------------------
    @router.post(
        "/suggestions/{suggestion_id}/dismiss", response_model=SuggestionItem, responses=_ERRORS
    )
    async def dismiss_suggestion(
        suggestion_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> SuggestionItem:
        """Dismiss a suggestion, for good and across every agent label.

        No version bump: nothing about the taxonomy changed. The row is kept
        because it *is* the decision — the re-proposal guard reads it so the same
        dense sink cluster is not proposed again on every run.
        """
        suggestion = await dismiss_topic_suggestion(suggestion_id, _owner_scope(user))

        _telemetry(
            "Recall Coverage Suggestion Dismissed API Endpoint Invoked",
            user,
            endpoint="POST /v1/recall-coverage/suggestions/{id}/dismiss",
            suggestion_id=str(suggestion_id),
        )
        return _suggestion_item(suggestion)

    # 9 --------------------------------------------------------------------
    @router.post(
        "/curated-questions",
        response_model=CuratedQuestionItem,
        status_code=201,
        responses=_ERRORS,
    )
    async def add_curated_question(
        payload: CreateCuratedQuestionRequest,
        user: User = Depends(get_authenticated_user),
    ) -> CuratedQuestionItem:
        """Add a curated question. **409** on a casefold-exact duplicate in the scope.

        Refused rather than merged so the writer learns the question is already
        covered; a silent merge would leave them believing they had added something.
        """
        question = await create_curated_question(
            user,
            payload.question_text,
            payload.scope,
            payload.agent_label,
            config=_config(),
        )

        _telemetry(
            "Recall Coverage Curated Question Added API Endpoint Invoked",
            user,
            endpoint="POST /v1/recall-coverage/curated-questions",
            scope=question.scope,
            agent_label=question.agent_label,
        )
        return _curated_item(question)

    # 10 -------------------------------------------------------------------
    @router.get("/curated-questions", response_model=list[CuratedQuestionItem], responses=_ERRORS)
    async def get_curated_questions(
        agent_label: Optional[str] = Query(
            default=None, description="Narrow the agent-scoped half to one label."
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[CuratedQuestionItem]:
        """The caller's agent-scoped rows plus every shared row, newest first."""
        questions = await list_curated_questions(user, agent_label, config=_config())
        return [_curated_item(question) for question in questions]

    # 11 -------------------------------------------------------------------
    @router.delete("/curated-questions/{question_id}", status_code=204, responses=_ERRORS)
    async def remove_curated_question(
        question_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> None:
        """Delete one curated question. 404 outside the caller's owner scope."""
        await delete_curated_question(user, question_id)

        _telemetry(
            "Recall Coverage Curated Question Deleted API Endpoint Invoked",
            user,
            endpoint="DELETE /v1/recall-coverage/curated-questions/{id}",
            question_id=str(question_id),
        )

    # 12 -------------------------------------------------------------------
    @router.get("/summary", response_model=BenchmarkMatrix, responses=_ERRORS)
    async def get_benchmark_summary(
        agent_labels: Optional[str] = Query(
            default=None,
            description="Comma-separated agent labels. Omitted means every label with a run.",
            examples=["claude-code,codex"],
        ),
        user: User = Depends(get_authenticated_user),
    ) -> BenchmarkMatrix:
        """The datasets x agents matrix, over **shared curated** rows only.

        A direct ``GROUP BY`` over each label's latest complete run. Restricted to
        the benchmark set because identical prompts across agents is the only reason
        two agents' numbers compare at all — an agent-scoped curated row is one
        person's list for one tool.
        """
        config = _config()
        labels = _requested_labels(agent_labels, user, config)

        latest = await latest_complete_runs(_owner_scope(user), labels or None)
        run_ids = {label: run.id for label, run in latest.items()}

        judge_score_max = build_params(None, config).judge_score_max
        cells = await benchmark_cells(run_ids)

        return BenchmarkMatrix(
            judge_score_max=judge_score_max,
            agent_labels=sorted(run_ids),
            cells=[_matrix_cell(cell, judge_score_max=judge_score_max) for cell in cells],
        )

    return router
