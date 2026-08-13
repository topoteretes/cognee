"""HTTP router for recall coverage — the ten routes mounted at ``/api/v1/coverage``.

Given an agent (a tool such as Claude Code or Codex), replay the questions it
recently asked, judge whether memory could answer them, and report where the gaps
are. Runs are always background: ``POST /api/v1/coverage`` inserts a ``pending``
row, schedules the coroutine and returns **202**; ``GET /runs/{run_id}`` is the
main read.

Four rules hold across every route here:

* **Response models are plain ``pydantic.BaseModel``, never ``OutDTO``.** ``OutDTO``
  sets ``alias_generator=to_camel`` and FastAPI serializes response models *by
  alias*, so an ``OutDTO`` here would emit ``topicId`` / ``coverageScore`` and break
  the agreed snake_case wire contract (see ``cognee/api/DTO.py``). Request bodies
  do use ``InDTO``, which accepts both cases — that is a convenience on the way in,
  and it does not touch what goes out.
* **The report is read from the run's frozen ``summary``**, never recomputed.
  Recomputing would let a deleted topic, or an owner losing access to a dataset,
  silently reshape a historical run and destroy the trend that stable topic ids
  exist to carry. Every read of it goes through ``.get``, so an unfinished run
  renders as "no numbers yet" instead of raising.
* **Authorisation is owner scope, not ``Depends(get_authenticated_user)`` alone.**
  That dependency falls back to the default user when authentication is optional,
  so every id-keyed route filters on **both** id and owner scope and 404s on a
  mismatch — never 403, which would confirm that another owner's row with that id
  exists. Errors are raised as ``CogneeApiError`` subclasses and turned into
  responses by the global handler in ``cognee/api/client.py``.
* **An unknown ``agent_label`` is a 422; a valid label with no traffic is an empty
  run, not an error.** A typo must not be indistinguishable from "this agent has
  asked nothing yet", which is a legitimate answer — and the expected one for
  prefix labels over history rows that predate ``Query.session_id``.
"""

from datetime import datetime
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.pipeline import start_recall_coverage_run
from cognee.modules.recall_coverage.repository import (
    CuratedQuestion,
    QuestionRecord,
    RunRecord,
    TopicRecord,
    create_curated_question,
    delete_curated_question,
    delete_topic,
    dismiss_topic_suggestion,
    get_run,
    list_curated_questions,
    list_runs,
    list_topics,
    load_run_questions,
    owner_scope_ids,
    parse_topic_id,
    topic_question_counts,
)
from cognee.modules.recall_coverage.suggest import create_topic_from_label
from cognee.modules.recall_coverage.types import SINK_TOPIC_LABEL, RunStatus
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger("recall_coverage")


# --- response models (plain BaseModel: the wire contract is snake_case) -------


class ErrorResponse(BaseModel):
    """Generic API error response, as produced by the global exception handler."""

    detail: str


class StartedRun(BaseModel):
    """The receipt for a scheduled run: enough to poll it, and nothing else.

    No counters and no score — the row is ``pending`` and has neither yet. The
    caller polls ``GET /runs/{run_id}`` with the id returned here.
    """

    run_id: str
    status: str
    agent_label: str
    created_at: Optional[datetime] = None


class RunListItem(BaseModel):
    """One row of the run history: identity, lifecycle and the headline number.

    ``memory_score`` comes from the run's frozen summary, which is what makes this
    list a trend rather than a log. It is null on a run that has not finished, and
    on a finished one where nothing was scored.

    A failed run appears here with ``status: "failed"`` and no score, but not with
    its reason: ``error`` belongs to the detail route, where the rest of the
    diagnosis lives. A history list is a trend, and one long exception message per
    row would be the widest column in it.
    """

    run_id: str
    status: str
    agent_label: str
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    question_count: int = 0
    memory_score: Optional[float] = None


class RunInfo(BaseModel):
    """The ``run`` block of the report: the two counters plus the frozen ``params``.

    ``recall_count`` counts raw recall rows in the window and ``question_count``
    the rows this run judged, so a window truncated by ``window_row_cap`` is
    visible in the report rather than only in the log. ``params`` is the snapshot
    the run executed under, which is what keeps it readable after the deployment's
    defaults move.

    ``error`` is why a failed run failed, and null on every other status. It lives
    here and not on the history list because this is the diagnosis route; without
    it a failed run's only explanation would be the server log.
    """

    run_id: str
    status: str
    agent_label: str
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    question_count: int = 0
    recall_count: int = 0
    params: Optional[dict] = None
    error: Optional[str] = None


class TopicScoreItem(BaseModel):
    """One topic's score. ``topic_id: null`` is the ``Uncategorized`` row.

    The sink is a member of this list rather than a block of its own: it is the
    residual of the same partition, and giving it its own shape made every reader
    special-case it. ``memory_score`` is null below the scored-row minimum. The
    sink reports its own average like any other row, but is left out of the run's
    ``memory_score``: it is a gap in the taxonomy, so averaging it into the
    headline would let an unplaceable question lower the score of a taxonomy that
    never claimed to cover it.
    """

    topic_id: Optional[str] = None
    topic: str = SINK_TOPIC_LABEL
    question_count: int = 0
    memory_score: Optional[float] = None


class SuggestedTopicItem(BaseModel):
    """One topic this run proposed: accept it by label, dismiss it by id.

    A per-run output, frozen into the summary: the review queue moves as the owner
    accepts and dismisses, and a historical run must keep showing what it proposed.

    The two halves of the review are deliberately keyed differently, which is why
    both a label and an id are here. Accepting goes through ``POST /topics`` with
    the **label**, because typing a name and clicking a proposed one are the same
    act and must mint the same kind of topic. Dismissing goes through
    ``POST /suggestions/{suggestion_id}/dismiss`` with the **id**, because it
    settles one specific proposed row rather than a name — and this list is the
    only place a client can learn that id. ``suggestion_id`` is optional only
    because every frozen field here is read defensively; a run written by this code
    always carries it.
    """

    suggestion_id: Optional[str] = None
    label: str
    question_count: int = 0


class QuestionRow(BaseModel):
    """One judged question row: ``(user_id, dataset_id, canonical text)``.

    Raw rows, deliberately ungrouped — grouping and filtering are the UI's job over
    the flat table. ``retrieval_context`` is stored but not returned: it is up to
    ``store_context_max_chars`` per row, and shipping it for every row would make
    the report an order of magnitude larger than the numbers anyone came for. A row
    the run could not judge simply carries ``coverage_score: null``.

    ``agent`` is the label resolved from *this row's* originating session, not the
    run's label: the default run covers ``all``, and narrowing one flat table to one
    agent is the entire point. A ``user_defined`` row nobody asked has no session
    and therefore no agent.

    ``answer`` is returned only on the **caller's own** rows. It is an LLM
    completion synthesized from the row user's private retrieval context — the
    same content ``retrieval_context`` deliberately withholds, in distilled form
    — so returning it on a teammate's row would hand the reader dataset content
    their ACL does not grant. Question text is shared (that is the spec'd team
    contract); answers are not.
    """

    question_id: str
    question: str
    coverage_score: Optional[int] = None
    relevance: int = 0
    topic: str = SINK_TOPIC_LABEL
    agent: Optional[str] = None
    user_id: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    answer: Optional[str] = None
    source: str
    first_asked_at: Optional[datetime] = None
    last_asked_at: Optional[datetime] = None


class CoverageReport(BaseModel):
    """The full report for one run: the frozen summary plus its raw rows."""

    run: RunInfo
    memory_score: Optional[float] = None
    topics: list[TopicScoreItem] = Field(default_factory=list)
    suggested_topics: list[SuggestedTopicItem] = Field(default_factory=list)
    questions: list[QuestionRow] = Field(default_factory=list)


class CreatedTopic(BaseModel):
    """A minted topic. The id is created here and then never moves.

    Stable because every later run assigns questions to it and the per-topic score
    trend is keyed on it — which is why accepting a suggestion goes through this
    same call rather than minting a second id for one theme.
    """

    topic_id: str
    topic: str
    created_at: Optional[datetime] = None


class TopicItem(CreatedTopic):
    """One stored topic, with how many questions the newest complete run put in it.

    A count as of the last finished run, not a lifetime total: a lifetime total
    would multiply by how often the owner happens to run coverage. A topic nothing
    has landed in — a freshly created one above all — honestly reports 0.
    """

    question_count: int = 0


class UserQuestionItem(BaseModel):
    """One question a human said memory should answer.

    One flat list per owner: no scope and no agent label, because what a person
    wants memory to answer is not a property of the tool that asks.
    """

    question_id: str
    question: str
    created_at: Optional[datetime] = None


# --- request models (InDTO: accepts snake_case and camelCase) ----------------


class StartRunRequest(InDTO):
    """``agent_label`` omitted defaults to ``all`` — every recall in the window."""

    agent_label: Optional[str] = Field(
        default=None,
        description=(
            "Agent to analyse: a label from the configured prefix map (including "
            "'ui'), or the literal 'api' / 'all'. Omitted means 'all'."
        ),
    )
    params: Optional[dict] = Field(
        default=None,
        description="Per-run overrides of the deployment's coverage parameters.",
    )


class CreateQuestionRequest(InDTO):
    """A question a human says memory should answer."""

    question: str = Field(description="The question, as a human would ask it.")


class CreateTopicRequest(InDTO):
    """A topic label: either typed by the owner, or a suggestion's label clicked."""

    topic: str = Field(description="The topic name, as it should read in the report.")


# --- mapping ------------------------------------------------------------------


def _run_error(run: RunRecord) -> Optional[str]:
    """A failed run's reason, out of the ``{"error": ...}`` summary it carries.

    ``fail_run`` stores the message in ``summary``, which ``_summary_of``
    rightly filters out (it is not a report); surfacing it here is what makes a
    failed run diagnosable through the API at all.
    """
    if run.status != RunStatus.FAILED.value or not isinstance(run.summary, dict):
        return None
    error = run.summary.get("error")
    return str(error) if error else None


def _summary_of(run: RunRecord) -> dict:
    """The run's frozen report, or an empty mapping.

    A pending run has no summary and a failed one carries ``{"error": ...}``, so
    anything without ``topics`` is not a report and is read as "no numbers yet".
    """
    summary = run.summary if isinstance(run.summary, dict) else {}
    return summary if "topics" in summary else {}


def _started_run(run: RunRecord) -> StartedRun:
    return StartedRun(
        run_id=str(run.id),
        status=run.status,
        agent_label=run.agent_label,
        created_at=run.created_at,
    )


def _run_list_item(run: RunRecord) -> RunListItem:
    return RunListItem(
        run_id=str(run.id),
        status=run.status,
        agent_label=run.agent_label,
        created_at=run.created_at,
        finished_at=run.finished_at,
        question_count=run.question_count,
        memory_score=_summary_of(run).get("memory_score"),
    )


def _run_info(run: RunRecord) -> RunInfo:
    return RunInfo(
        run_id=str(run.id),
        status=run.status,
        agent_label=run.agent_label,
        created_at=run.created_at,
        finished_at=run.finished_at,
        question_count=run.question_count,
        recall_count=run.recall_count,
        params=run.params,
        error=_run_error(run),
    )


def _topic_score(cell: Mapping[str, Any]) -> TopicScoreItem:
    """One frozen ``topics[]`` entry, read key by key.

    Field by field through ``.get`` rather than ``TopicScoreItem(**cell)``: the
    summary is whatever the run that wrote it froze, and a report must render a
    row from an older payload rather than 500 on a key that has since been renamed.
    """
    topic_id = cell.get("topic_id")
    return TopicScoreItem(
        topic_id=None if topic_id is None else str(topic_id),
        topic=str(cell.get("topic") or SINK_TOPIC_LABEL),
        question_count=int(cell.get("question_count") or 0),
        memory_score=cell.get("memory_score"),
    )


def _suggested_topic(cell: Mapping[str, Any]) -> SuggestedTopicItem:
    suggestion_id = cell.get("suggestion_id")
    return SuggestedTopicItem(
        suggestion_id=None if suggestion_id is None else str(suggestion_id),
        label=str(cell.get("label") or ""),
        question_count=int(cell.get("question_count") or 0),
    )


def _topic_labels(summary: Mapping[str, Any]) -> dict[str, str]:
    """Topic id -> label, straight out of the frozen summary.

    The labels come from the summary rather than from a join on
    ``recall_coverage_topics`` precisely because a topic may since have been
    deleted, and the report must still name what it measured.
    """
    return {
        str(topic.get("topic_id")): str(topic.get("topic") or SINK_TOPIC_LABEL)
        for topic in summary.get("topics", [])
        if topic.get("topic_id")
    }


def _question_row(
    record: QuestionRecord, labels: Mapping[str, str], *, viewer_id: UUID
) -> QuestionRow:
    return QuestionRow(
        question_id=str(record.id),
        question=record.question,
        coverage_score=record.coverage_score,
        relevance=record.relevance,
        # No stored topic_id means Uncategorized; an id the summary does not name
        # means the same thing to a reader, and inventing a label would be worse.
        topic=SINK_TOPIC_LABEL
        if record.topic_id is None
        else labels.get(str(record.topic_id), SINK_TOPIC_LABEL),
        agent=record.agent_label,
        user_id=str(record.user_id),
        dataset_id=None if record.dataset_id is None else str(record.dataset_id),
        dataset_name=record.dataset_name,
        # Withheld on other users' rows: the answer is distilled from the row
        # user's private retrieval context. See the QuestionRow docstring.
        answer=record.answer if record.user_id == viewer_id else None,
        source=record.source,
        first_asked_at=record.first_asked_at,
        last_asked_at=record.last_asked_at,
    )


def _report(
    run: RunRecord, questions: Sequence[QuestionRecord], *, viewer_id: UUID
) -> CoverageReport:
    summary = _summary_of(run)
    labels = _topic_labels(summary)

    return CoverageReport(
        run=_run_info(run),
        memory_score=summary.get("memory_score"),
        topics=[_topic_score(cell) for cell in summary.get("topics", [])],
        suggested_topics=[_suggested_topic(cell) for cell in summary.get("suggested_topics", [])],
        questions=[_question_row(record, labels, viewer_id=viewer_id) for record in questions],
    )


def _created_topic(topic: TopicRecord) -> CreatedTopic:
    return CreatedTopic(
        topic_id=str(topic.id),
        topic=topic.label,
        created_at=topic.created_at,
    )


def _topic_item(topic: TopicRecord, counts: Mapping[UUID, int]) -> TopicItem:
    return TopicItem(
        topic_id=str(topic.id),
        topic=topic.label,
        created_at=topic.created_at,
        question_count=counts.get(topic.id, 0),
    )


def _question_item(question: CuratedQuestion) -> UserQuestionItem:
    return UserQuestionItem(
        question_id=str(question.id),
        question=question.question,
        created_at=question.created_at,
    )


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
    @router.post("", response_model=StartedRun, status_code=202, responses=_ERRORS)
    async def start_run(
        payload: Optional[StartRunRequest] = None,
        user: User = Depends(get_authenticated_user),
    ) -> StartedRun:
        """Start a coverage run in the background and return its ``pending`` row.

        Always background — a run replays and judges every question row, which is
        minutes of LLM calls. **409** when a run for this ``(owner, agent_label)``
        is already pending or running: overlapping runs would multiply the cost and
        race on the same taxonomy. The guard is bounded by
        ``run_stale_after_seconds``, because status is not liveness.
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
            endpoint="POST /v1/coverage",
            agent_label=run.agent_label,
            run_id=str(run.id),
        )
        return _started_run(run)

    # 2 --------------------------------------------------------------------
    @router.get("/runs", response_model=list[RunListItem], responses=_ERRORS)
    async def list_coverage_runs(
        agent_label: Optional[str] = Query(
            default=None, description="Narrow the list to one agent label."
        ),
        limit: Optional[int] = Query(
            default=None,
            ge=1,
            description="Newest N runs. Defaults to runs_list_default_limit.",
        ),
        user: User = Depends(get_authenticated_user),
    ) -> list[RunListItem]:
        """The caller's runs, newest first — the score trend over time."""
        config = _config()
        label = (
            resolve_agent_scope(agent_label, user=user, config=config).label
            if agent_label
            else None
        )

        runs = await list_runs(
            owner_scope_ids(user),
            label,
            limit=limit if limit is not None else config.runs_list_default_limit,
        )
        return [_run_list_item(run) for run in runs]

    # 3 --------------------------------------------------------------------
    @router.get("/runs/{run_id}", response_model=CoverageReport, responses=_ERRORS)
    async def get_coverage_report(
        run_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> CoverageReport:
        """The full report: the run's frozen summary plus its raw question rows.

        The main read endpoint. Rows come back ungrouped, user-defined rows pinned
        above observed ones and then most-asked first; grouping and filtering are
        the UI's job over one flat table.
        """
        run = await get_run(run_id, owner_scope_ids(user))
        questions = await load_run_questions(run.id)

        _telemetry(
            "Recall Coverage Report Read API Endpoint Invoked",
            user,
            endpoint="GET /v1/coverage/runs/{run_id}",
            agent_label=run.agent_label,
            run_id=str(run.id),
        )
        return _report(run, questions, viewer_id=user.id)

    # 4 --------------------------------------------------------------------
    @router.post("/questions", response_model=UserQuestionItem, status_code=201, responses=_ERRORS)
    async def add_user_question(
        payload: CreateQuestionRequest,
        user: User = Depends(get_authenticated_user),
    ) -> UserQuestionItem:
        """Add a question memory should answer. **409** on a casefold-exact duplicate.

        Refused rather than merged so the writer learns the question is already
        covered; a silent merge would leave them believing they had added something.
        **422** once the list holds ``max_curated_questions`` rows — a full list
        makes the value unprocessable rather than the state conflicting — because
        every row costs a replay, a judge call and an answer completion on every run.
        """
        question = await create_curated_question(user, payload.question, config=_config())

        _telemetry(
            "Recall Coverage Question Added API Endpoint Invoked",
            user,
            endpoint="POST /v1/coverage/questions",
            question_id=str(question.id),
        )
        return _question_item(question)

    # 5 --------------------------------------------------------------------
    @router.get("/questions", response_model=list[UserQuestionItem], responses=_ERRORS)
    async def get_user_questions(
        user: User = Depends(get_authenticated_user),
    ) -> list[UserQuestionItem]:
        """The caller's own questions, newest first. One flat list, no label filter."""
        questions = await list_curated_questions(user)
        return [_question_item(question) for question in questions]

    # 6 --------------------------------------------------------------------
    @router.delete("/questions/{question_id}", status_code=204, responses=_ERRORS)
    async def remove_user_question(
        question_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> None:
        """Delete one of the caller's questions. 404 outside their own list."""
        await delete_curated_question(user, question_id)

        _telemetry(
            "Recall Coverage Question Deleted API Endpoint Invoked",
            user,
            endpoint="DELETE /v1/coverage/questions/{id}",
            question_id=str(question_id),
        )

    # 7 --------------------------------------------------------------------
    @router.post("/topics", response_model=CreatedTopic, status_code=201, responses=_ERRORS)
    async def add_topic(
        payload: CreateTopicRequest,
        user: User = Depends(get_authenticated_user),
    ) -> CreatedTopic:
        """Create a topic — and accept a suggestion, when the label matches one.

        One route for both of the UI's flows, because from the owner's side they are
        one act: they type a name, or they click a proposed one. If the posted label
        matches a pending suggestion within ``suggestion_dedup_threshold``, that
        suggestion's centroid is copied verbatim and the suggestion is marked
        accepted, which is what stops the theme being proposed again next run.
        Otherwise the label is embedded and *is* the new topic's centroid, so it
        only attracts questions worded like its name until traffic drifts towards it.

        **422** on a blank label, **409** on a casefold-exact duplicate of an active
        one: two topics with near-identical centroids cannot be separated by the
        assignment margin rule, so every question about that theme would land in
        ``Uncategorized`` instead — a duplicate label disables a topic rather than
        adding one. **500** under ``MOCK_EMBEDDING=true``, which cannot produce a
        centroid that means anything.
        """
        topic, accepted = await create_topic_from_label(user.id, payload.topic, config=_config())

        _telemetry(
            "Recall Coverage Topic Created API Endpoint Invoked",
            user,
            endpoint="POST /v1/coverage/topics",
            topic_id=str(topic.id),
            accepted_suggestion_id=None if accepted is None else str(accepted.id),
        )
        return _created_topic(topic)

    # 8 --------------------------------------------------------------------
    @router.get("/topics", response_model=list[TopicItem], responses=_ERRORS)
    async def get_topics(
        user: User = Depends(get_authenticated_user),
    ) -> list[TopicItem]:
        """The owner's taxonomy, oldest first, with each topic's question count.

        Owner-scoped and with **no ``agent_label`` parameter**: one taxonomy serves
        all of an owner's agents, which is what makes "Codex 4.2 on Billing, Claude
        Code 2.1 on Billing" a sentence at all. Soft-deleted topics are never
        listed — the row survives only so a historical run can still resolve the
        topic id on its own question rows.

        Pending suggestions are not here: they are a per-run output and travel in
        the run report.
        """
        owners = owner_scope_ids(user)
        topics = await list_topics(owners)
        counts = await topic_question_counts(owners)

        return [_topic_item(topic, counts) for topic in topics]

    # 9 --------------------------------------------------------------------
    @router.delete("/topics/{topic_id}", status_code=204, responses=_ERRORS)
    async def remove_topic(
        topic_id: str,
        user: User = Depends(get_authenticated_user),
    ) -> None:
        """Soft-delete a topic. Idempotent, and nothing to report back.

        The topic's questions are never deleted — they fall back to
        ``Uncategorized`` on the next run, which is exactly the signal the
        ``Uncategorized`` row exists to give. The path parameter is parsed here
        rather than by FastAPI so an unparseable id is a 404 (it names nothing)
        instead of a 422.
        """
        parsed = parse_topic_id(topic_id)
        await delete_topic(parsed, owner_scope_ids(user))

        _telemetry(
            "Recall Coverage Topic Deleted API Endpoint Invoked",
            user,
            endpoint="DELETE /v1/coverage/topics/{topic_id}",
            topic_id=str(parsed),
        )

    # 10 -------------------------------------------------------------------
    @router.post("/suggestions/{suggestion_id}/dismiss", status_code=204, responses=_ERRORS)
    async def dismiss_suggestion(
        suggestion_id: UUID,
        user: User = Depends(get_authenticated_user),
    ) -> None:
        """Dismiss a suggestion, for good and across every agent label.

        The id comes from ``suggested_topics[].suggestion_id`` in a run report,
        which is the only place it is published — the accept half of the same
        review is keyed on the label instead (``POST /topics``).

        **409** on an already decided one. The row is kept because it *is* the
        decision — the re-proposal guard reads it so the same dense cluster is not
        proposed again on every run.
        """
        await dismiss_topic_suggestion(suggestion_id, owner_scope_ids(user))

        _telemetry(
            "Recall Coverage Suggestion Dismissed API Endpoint Invoked",
            user,
            endpoint="POST /v1/coverage/suggestions/{id}/dismiss",
            suggestion_id=str(suggestion_id),
        )

    return router
