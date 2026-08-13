"""Run one recall-coverage analysis end to end.

Four phases, in this order, over one ``recall_coverage_runs`` row this module owns:

1. **Fetch and dedup** — the window out of ``queries``, collapsed into distinct
   asks, the owner's user-defined questions appended, embedded once, then deduped
   per ``(user_id, dataset_id)`` partition.
2. **Assign and suggest** — every question row to at most one of the owner's
   topics or to ``Uncategorized``, then dense unmatched clusters proposed as
   ``pending`` suggestions.
3. **Replay and judge** — one search per row *as that row's own user*, then the
   coverage score and, only above zero, an answer generated from the same context.
4. **Aggregate and persist** — the rows and one frozen ``summary``, in a single
   transaction.

Three structural decisions, each of which the obvious alternative gets wrong:

* **No ``run_pipeline``, no ``Task`` graph, no ``PipelineRun``.** ``run_pipeline``
  iterates over authorized datasets and ``PipelineRun.dataset_id`` is scalar, so a
  tenant-wide run would mint one ``PipelineRun`` per dataset — and, worse, take
  ``get_dataset_lock`` on every one of them, blocking ``add()`` and ``cognify()``
  on all N datasets for the minutes a run takes. So this owns its own id and its
  own status, driven by a plain background task.
* **Phase 2 runs before phase 3.** A stale topic centroid fails the run
  (``EmbeddingFingerprintMismatchError``), and it must fail *before* the replay
  and judge calls, which are the only expensive part. Ordering it the other way
  round would bill a full run to discover a configuration problem.
* **The background task is anchored in a module-level set.** The event loop keeps
  only a weak reference to a task, so an unanchored run can be garbage-collected
  mid-flight and silently stop — leaving a row stuck at ``running`` forever.

Failure policy: anything that escapes a phase marks the run ``failed`` with the
message in ``summary`` and re-raises, so the caller (or the task's exception
handler) still sees it. Per-row failures — an unreadable dataset, a missing user,
a judge that gave up — are *not* run failures: those rows carry an ``error`` and
NULL scores, because one broken row must not throw away a hundred judged ones.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.recall_coverage.agent_scope import classify_session, resolve_agent_scope
from cognee.modules.recall_coverage.aggregate import (
    SuggestedTopic,
    build_rows,
    run_counters,
    summarize,
)
from cognee.modules.recall_coverage.assign import assign_topics, canonical_matrix
from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.dedup import Ask, collapse_asks, dedup_asks
from cognee.modules.recall_coverage.embedding import embed_normalized, engine_fingerprint
from cognee.modules.recall_coverage.exceptions import (
    CoverageRunInFlightError,
    InvalidCoverageParamsError,
)
from cognee.modules.recall_coverage.judge import judge_rows, preload_judge_prompts
from cognee.modules.recall_coverage.replay import (
    ReplayUserCache,
    error_text,
    replay_questions,
)
from cognee.modules.recall_coverage.repository import (
    RunRecord,
    SuggestionRecord,
    create_run,
    fail_run,
    load_active_topics,
    load_curated_asks,
    mark_run_running,
    persist_run_results,
    runs_in_flight,
    visible_user_ids,
)
from cognee.modules.recall_coverage.suggest import suggest_topics
from cognee.modules.recall_coverage.types import AgentScope, CoverageParams
from cognee.modules.search.operations import count_queries, get_queries
from cognee.modules.users.methods import get_user
from cognee.shared.logging_utils import get_logger

logger = get_logger("recall_coverage")

# Strong references to in-flight background runs. The event loop holds only a
# weak reference to a task, so without this a run can be collected mid-flight and
# its row would sit at "running" for ever. Mirrors the sync module's anchor set.
_BACKGROUND_RUN_TASKS: set[asyncio.Task] = set()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_params(
    overrides: Optional[Mapping[str, Any]] = None,
    config: Optional[RecallCoverageConfig] = None,
) -> CoverageParams:
    """Snapshot the deployment defaults with per-request overrides applied.

    ``CoverageParams`` forbids extra keys, so a typo'd request parameter raises
    instead of being silently dropped — a request that looked accepted while
    running under a different threshold is worse than a rejected one. That raise
    is translated here into a 422, because it is the caller's mistake and not a
    server fault.
    """
    try:
        return CoverageParams.from_config(config, **dict(overrides or {}))
    except Exception as error:
        raise InvalidCoverageParamsError(message=f"Invalid recall coverage parameters: {error}")


async def start_recall_coverage_run(
    user: Any,
    agent_label: Optional[str] = None,
    *,
    params: Optional[Mapping[str, Any]] = None,
    config: Optional[RecallCoverageConfig] = None,
) -> RunRecord:
    """Validate, guard, insert a ``pending`` run and schedule it. Returns immediately.

    The row is written **before** the coroutine is scheduled, so it is what the
    in-flight guard sees and what the 202 reports: a run that existed only inside
    a task would be invisible to the guard, and two requests a second apart would
    both replay and judge the same window at full LLM cost.

    The guard is per ``(owner, agent_label)``. Per owner because a run is started
    and paid for by one caller, so a teammate's run must not block theirs; per
    label because analysing Codex and Claude Code at the same time is legitimate —
    they are different windows over the same taxonomy. And it is bounded by
    ``run_stale_after_seconds``, because status is not liveness: the background
    task lives in one process, so a pod rescheduled mid-run would otherwise leave
    ``running`` on the row for ever and 409 every later run for that pair.
    """
    if config is None:
        config = get_recall_coverage_config()

    scope = resolve_agent_scope(agent_label, user=user, config=config)
    coverage_params = build_params(params, config)

    in_flight = await runs_in_flight(
        user.id, scope.label, stale_after_seconds=config.run_stale_after_seconds
    )
    if in_flight:
        raise CoverageRunInFlightError(
            message=(
                f"A recall coverage run for {scope.label!r} is already "
                f"{in_flight[0].status} (run {in_flight[0].id})."
            )
        )

    run = await create_run(user.id, scope.label, params=coverage_params)

    schedule_recall_coverage_run(run.id, scope, user.id, params=coverage_params, config=config)
    return run


def _release_run_task(task: asyncio.Task) -> None:
    """Unanchor a finished run and retrieve its exception.

    ``run_recall_coverage`` has already logged the failure and marked the row
    ``failed`` by the time this runs, but an exception nobody retrieves also makes
    asyncio emit its own "Task exception was never retrieved" when the task is
    collected — a second, context-free traceback for a failure that was handled.
    """
    _BACKGROUND_RUN_TASKS.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.debug("recall_coverage: background run task ended with %r", error)


def schedule_recall_coverage_run(
    run_id: UUID,
    scope: AgentScope,
    user_id: UUID,
    *,
    params: CoverageParams,
    config: Optional[RecallCoverageConfig] = None,
) -> asyncio.Task:
    """Fire the run off in the background, anchored against garbage collection."""
    task = asyncio.create_task(
        run_recall_coverage(run_id, scope, user_id, params=params, config=config)
    )
    _BACKGROUND_RUN_TASKS.add(task)
    task.add_done_callback(_release_run_task)
    return task


async def _observed_asks(
    scope: AgentScope, *, params: CoverageParams, user_ids: Sequence[UUID]
) -> tuple[list[Ask], int]:
    """Phase 1: the window, collapsed into distinct asks.

    Returns the (truncated) asks plus ``recall_count``, the only phase-1 counter the
    run row reports. ``user_ids`` is the boundary of the run — the users the caller
    may analyse (their tenant's members, or themselves; see
    :func:`cognee.modules.recall_coverage.repository.visible_user_ids`). It is
    required, not optional: an unbounded window reads strangers' question text on
    any deployment whose relational database holds more than one tenant.

    The fetch is bounded by ``window_row_cap`` in SQL. When the cap is hit,
    ``recall_count`` is corrected by a ``COUNT(*)`` over the same filters so it
    stays a statement about the whole window rather than about the sample, and the
    log line says so. ``max_questions`` bounds the expensive part — the embedding
    and the dedup matmul — by truncating *after* the collapse, newest first.
    """
    since = _utc_now() - timedelta(days=params.max_age_days)

    rows = await get_queries(
        limit=params.window_row_cap,
        since=since,
        query_types=params.query_types,
        session_scope=scope,
        user_ids=user_ids,
    )

    collapsed = collapse_asks(
        rows,
        fanout_window_seconds=params.fanout_window_seconds,
        retry_cooldown_seconds=params.retry_cooldown_seconds,
        max_questions=params.max_questions,
    )

    recall_count = collapsed.recall_count
    if len(rows) >= params.window_row_cap:
        recall_count = await count_queries(
            since=since,
            query_types=params.query_types,
            session_scope=scope,
            user_ids=user_ids,
        )
        logger.warning(
            "recall_coverage: window truncated at %s of %s rows; the asks describe the "
            "newest %s rows only",
            params.window_row_cap,
            recall_count,
            params.window_row_cap,
        )

    logger.debug(
        "recall_coverage: %s window rows collapsed into %s asks (%s retries swallowed)",
        recall_count,
        collapsed.distinct_ask_count,
        collapsed.collapsed_retry_count,
    )
    return collapsed.asks, recall_count


async def run_recall_coverage(
    run_id: UUID,
    scope: AgentScope,
    user_id: UUID,
    *,
    params: CoverageParams,
    config: Optional[RecallCoverageConfig] = None,
) -> RunRecord:
    """Execute one coverage run against ``run_id`` and return the completed row.

    ``user_id`` rather than a ``User``: this runs in a background task that
    outlives the request, and a ``User`` handed across that boundary raises
    ``DetachedInstanceError`` on the first relationship access, so the user is
    re-loaded here (the same reason ``get_authenticated_user`` re-fetches).

    The caller is the run's owner and therefore the owner of the taxonomy the run
    is scored against and of the user-defined questions it includes. The *questions*
    are the tenant's: a run covers every user in the caller's tenant — and only
    them, via :func:`visible_user_ids`, because the relational database itself is
    not a tenant boundary (OSS deployments share one).

    ``config`` is threaded through beside ``scope`` because per-row attribution
    needs the prefix map: each row's ``agent_label`` comes from **its own** session
    id, not from ``scope.label``. The default run is ``all``, and narrowing one flat
    table to one agent is the whole point of the column.
    """
    if config is None:
        config = get_recall_coverage_config()

    try:
        await mark_run_running(run_id)

        user = await get_user(user_id)
        owner_id = user_id

        # Fail-before-spend, like the fingerprint check: a missing judge prompt
        # file must surface here, not per-row after the replay was paid for.
        preload_judge_prompts()

        user_ids = await visible_user_ids(user)

        asks, recall_count = await _observed_asks(scope, params=params, user_ids=user_ids)

        # Phase 1: the owner's user-defined questions, appended after the
        # truncation so a human's list never displaces observed traffic, and
        # replicated into one partition per dataset the caller can read. The
        # replication is bounded by the same ``max_questions`` budget the observed
        # rows already obey — these are the one input the window caps don't touch,
        # and N_questions x N_datasets multiplies replay and judge cost.
        datasets = await get_authorized_existing_datasets(
            datasets=None, permission_type="read", user=user
        )
        dataset_names: dict[UUID, str] = {dataset.id: dataset.name for dataset in datasets}

        curated_rows = await load_curated_asks(
            user, dataset_ids=[dataset.id for dataset in datasets]
        )
        if len(curated_rows) > params.max_questions:
            logger.warning(
                "recall_coverage: %s user-defined question rows exceed the max_questions "
                "budget of %s; rows beyond the budget were dropped",
                len(curated_rows),
                params.max_questions,
            )
            curated_rows = curated_rows[: params.max_questions]
        asks = list(asks) + curated_rows

        if not asks:
            # An empty window is a complete run with memory_score null and no
            # questions — "nothing asked yet", not a failure. Expected for an
            # agent that has not asked anything in the window.
            return await _persist_empty(run_id, params=params, recall_count=recall_count)

        # Phase 1, continued.
        engine = get_embedding_engine()
        fingerprint = engine_fingerprint(engine)
        normalized = await embed_normalized(engine, [ask.text for ask in asks])

        dedup = dedup_asks(asks, normalized, dedup_threshold=params.dedup_threshold)
        questions = dedup.questions

        # Phase 2. Deliberately ahead of the replay: a fingerprint mismatch fails
        # the run, and it must do so before anything is spent on LLM calls.
        topics = await load_active_topics(owner_id)
        question_vectors = canonical_matrix(questions, normalized)
        assignment = assign_topics(
            question_vectors,
            topics,
            fingerprint=fingerprint,
            assignment_threshold=params.assignment_threshold,
            assignment_margin=params.assignment_margin,
        )

        suggestions = await _suggest_from_sink(
            owner_id,
            questions,
            question_vectors,
            assignment.sink_indices,
            params=params,
            fingerprint=fingerprint,
            run_id=run_id,
            agent_label=scope.label,
        )

        # Phase 3.
        replayed = await replay_questions(questions, params=params, user_cache=ReplayUserCache())
        judged = await judge_rows(questions, replayed, params=params)

        # Phase 4.
        rows = build_rows(
            questions,
            assignment.assignments,
            replayed,
            judged,
            store_context_max_chars=params.store_context_max_chars,
            dataset_names=dataset_names,
            agent_label_of=lambda session_id: classify_session(session_id, config),
        )
        summary = summarize(
            rows,
            params=params,
            suggested_topics=_suggested_topics(suggestions),
        )
        counters = run_counters(rows, recall_count=recall_count)

        return await persist_run_results(run_id, rows, summary, counters, params=params)
    except Exception as error:
        logger.error("recall_coverage: run %s failed: %s", run_id, error, exc_info=True)
        # Bounded and class-prefixed, like the per-row errors: this string is
        # returned by the API as RunInfo.error. The log line above keeps the
        # full traceback.
        await fail_run(run_id, error_text(error))
        raise


async def _persist_empty(run_id: UUID, *, params: CoverageParams, recall_count: int) -> RunRecord:
    """Complete a run that had nothing to judge, ``recall_count`` intact.

    ``recall_count`` still matters here and ``question_count`` is 0: a window that
    held recalls but produced no question rows is a different situation from a
    window that held nothing, and the run row is the only place that survives to
    say which happened.
    """
    summary = summarize([], params=params)
    counters = run_counters([], recall_count=recall_count)
    logger.debug("recall_coverage: run %s completed over an empty window", run_id)
    return await persist_run_results(run_id, [], summary, counters, params=params)


def _suggested_topics(suggestions: Sequence[SuggestionRecord]) -> list[SuggestedTopic]:
    """This run's pending suggestions, as the frozen report carries them.

    ``cohesion`` is dropped here rather than at the edge: it orders the candidates
    inside :mod:`cognee.modules.recall_coverage.suggest` and says nothing about
    memory, so it must not reach a report a reader could mistake it for a score in.
    The id is kept, because the report is the only place the dismiss route's
    ``suggestion_id`` can come from.
    """
    return [
        SuggestedTopic(
            suggestion_id=suggestion.id,
            label=suggestion.label,
            question_count=suggestion.question_count,
        )
        for suggestion in suggestions
    ]


async def _suggest_from_sink(
    owner_id: UUID,
    questions: Sequence[Any],
    question_vectors: Any,
    sink_indices: Sequence[int],
    *,
    params: CoverageParams,
    fingerprint: Any,
    run_id: UUID,
    agent_label: str,
) -> list[SuggestionRecord]:
    """Propose topics from dense unmatched clusters; return what was written.

    The written suggestions are what the run reports, so they are returned rather
    than re-read: a suggestion is a per-run output, and the review queue moves as
    the owner accepts and dismisses.
    """
    if not sink_indices:
        return []

    sink_texts = [questions[index].text for index in sink_indices]
    sink_vectors = question_vectors[list(sink_indices)]

    suggestions = await suggest_topics(
        owner_id,
        sink_texts,
        sink_vectors,
        params=params,
        fingerprint=fingerprint,
        run_id=run_id,
        agent_label=agent_label,
    )
    logger.debug(
        "recall_coverage: %s unmatched questions produced %s suggestions",
        len(sink_indices),
        len(suggestions),
    )
    return suggestions


__all__ = [
    "build_params",
    "run_recall_coverage",
    "schedule_recall_coverage_run",
    "start_recall_coverage_run",
]
