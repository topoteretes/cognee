"""Run one recall-coverage analysis end to end.

Spec section 2. Four phases, in this order, over one ``recall_coverage_runs`` row
this module owns:

1. **Fetch and dedup** — the window out of ``queries``, collapsed into distinct
   asks, curated questions appended, embedded once, deduped per
   ``(user_id, dataset_id)`` partition, then stamped with cross-partition
   ``question_group_id``s.
2. **Assign and suggest** — every question row to at most one of the owner's
   topics or to the sink, then dense sink clusters proposed as ``pending``
   suggestions.
3. **Replay and judge** — one search per row *as that row's own user*, then the
   coverage score and (only above zero) an answer and an ``answered`` verdict.
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
from cognee.modules.recall_coverage.agent_scope import resolve_agent_scope
from cognee.modules.recall_coverage.aggregate import (
    build_rows,
    run_counters,
    summarize,
)
from cognee.modules.recall_coverage.assign import assign_topics, canonical_matrix
from cognee.modules.recall_coverage.config import (
    RecallCoverageConfig,
    get_recall_coverage_config,
)
from cognee.modules.recall_coverage.dedup import (
    Ask,
    assign_question_groups,
    collapse_asks,
    dedup_asks,
    group_by_similarity,
)
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
    create_run,
    curated_asks,
    current_taxonomy_version,
    fail_run,
    load_active_topics,
    load_curated_questions_for_scope,
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

    run = await create_run(
        user.id,
        scope.label,
        params=coverage_params,
        taxonomy_version=await current_taxonomy_version(user.id),
    )

    schedule_recall_coverage_run(run.id, scope, user.id, params=coverage_params)
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
) -> asyncio.Task:
    """Fire the run off in the background, anchored against garbage collection."""
    task = asyncio.create_task(run_recall_coverage(run_id, scope, user_id, params=params))
    _BACKGROUND_RUN_TASKS.add(task)
    task.add_done_callback(_release_run_task)
    return task


async def _observed_asks(
    scope: AgentScope, *, params: CoverageParams, user_ids: Sequence[UUID]
) -> tuple[list[Ask], int, int, int]:
    """Phase 1 steps 1-2: the window, collapsed into distinct asks.

    Returns the (truncated) asks plus the three counters the run row reports.
    ``user_ids`` is the boundary of the run — the users the caller may analyse
    (their tenant's members, or themselves; see
    :func:`cognee.modules.recall_coverage.repository.visible_user_ids`). It is
    required, not optional: an unbounded window reads strangers' question text on
    any deployment whose relational database holds more than one tenant.

    The fetch is bounded by ``window_row_cap`` in SQL. When the cap is hit,
    ``recall_row_count`` is corrected by a ``COUNT(*)`` over the same filters so
    it stays a statement about the whole window; ``distinct_ask_count`` and
    ``collapsed_retry_count`` then describe the newest ``window_row_cap`` rows,
    which the log line says out loud. ``max_questions`` bounds the expensive part
    — the embedding and the dedup matmul — by truncating *after* the collapse,
    newest first.
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

    recall_row_count = collapsed.recall_row_count
    if len(rows) >= params.window_row_cap:
        recall_row_count = await count_queries(
            since=since,
            query_types=params.query_types,
            session_scope=scope,
            user_ids=user_ids,
        )
        logger.warning(
            "recall_coverage: window truncated at %s of %s rows; ask and retry "
            "counters describe the newest %s rows only",
            params.window_row_cap,
            recall_row_count,
            params.window_row_cap,
        )

    logger.debug(
        "recall_coverage: %s window rows collapsed into %s asks (%s retries swallowed)",
        recall_row_count,
        collapsed.distinct_ask_count,
        collapsed.collapsed_retry_count,
    )
    return (
        collapsed.asks,
        recall_row_count,
        collapsed.distinct_ask_count,
        collapsed.collapsed_retry_count,
    )


async def run_recall_coverage(
    run_id: UUID,
    scope: AgentScope,
    user_id: UUID,
    *,
    params: CoverageParams,
) -> RunRecord:
    """Execute one coverage run against ``run_id`` and return the completed row.

    ``user_id`` rather than a ``User``: this runs in a background task that
    outlives the request, and a ``User`` handed across that boundary raises
    ``DetachedInstanceError`` on the first relationship access, so the user is
    re-loaded here (the same reason ``get_authenticated_user`` re-fetches).

    The caller is the run's owner and therefore the owner of the taxonomy the run
    is scored against and of the curated questions it includes. The *questions*
    are the tenant's: a run covers every user in the caller's tenant — and only
    them, via :func:`visible_user_ids`, because the relational database itself is
    not a tenant boundary (OSS deployments share one).
    """
    try:
        await mark_run_running(run_id)

        user = await get_user(user_id)
        owner_id = user_id

        # Fail-before-spend, like the fingerprint check: a missing judge prompt
        # file must surface here, not per-row after the replay was paid for.
        preload_judge_prompts()

        user_ids = await visible_user_ids(user)

        asks, recall_row_count, distinct_ask_count, collapsed_retry_count = await _observed_asks(
            scope, params=params, user_ids=user_ids
        )

        # Phase 1 step 3: curated questions, appended after the truncation so a
        # human's list never displaces observed traffic, and replicated into one
        # partition per dataset the caller can read. The replication is bounded
        # by the same ``max_questions`` budget the observed rows already obey —
        # curated rows are the one input the window caps don't touch, and
        # N_curated x N_datasets multiplies replay and judge cost. Shared rows
        # survive truncation first: they are the benchmark set, and dropping one
        # silently un-compares every agent scored on it.
        datasets = await get_authorized_existing_datasets(
            datasets=None, permission_type="read", user=user
        )
        dataset_names: dict[UUID, str] = {dataset.id: dataset.name for dataset in datasets}

        curated = await load_curated_questions_for_scope(user, scope)
        curated = sorted(curated, key=lambda question: not question.is_shared)
        shared_curated_ids = {question.id for question in curated if question.is_shared}
        curated_rows = curated_asks(
            curated, user_id=user.id, dataset_ids=[dataset.id for dataset in datasets]
        )
        if len(curated_rows) > params.max_questions:
            logger.warning(
                "recall_coverage: %s curated question rows exceed the max_questions "
                "budget of %s; rows beyond the budget were dropped, shared benchmark "
                "rows kept first",
                len(curated_rows),
                params.max_questions,
            )
            curated_rows = curated_rows[: params.max_questions]
        asks = list(asks) + curated_rows

        if not asks:
            # An empty window is a complete run with overall_score null and no
            # questions — "nothing asked yet", not a failure. Expected for an
            # agent that has not asked anything in the window, and for prefix
            # labels over history that predates Query.session_id.
            return await _persist_empty(
                run_id,
                params=params,
                owner_id=owner_id,
                recall_row_count=recall_row_count,
                distinct_ask_count=distinct_ask_count,
                collapsed_retry_count=collapsed_retry_count,
            )

        # Phase 1 steps 4-6.
        engine = get_embedding_engine()
        fingerprint = engine_fingerprint(engine)
        normalized = await embed_normalized(engine, [ask.text for ask in asks])

        dedup = dedup_asks(asks, normalized, dedup_threshold=params.dedup_threshold)
        questions = dedup.questions
        assign_question_groups(questions, normalized, dedup_threshold=params.dedup_threshold)

        # Phase 2. Deliberately ahead of the replay: a fingerprint mismatch fails
        # the run, and it must do so before anything is spent on LLM calls.
        taxonomy_version = await current_taxonomy_version(owner_id)
        topics = await load_active_topics(owner_id)
        question_vectors = canonical_matrix(questions, normalized)
        assignment = assign_topics(
            question_vectors,
            topics,
            fingerprint=fingerprint,
            assignment_threshold=params.assignment_threshold,
            assignment_margin=params.assignment_margin,
        )

        sink_cluster_sizes = await _suggest_from_sink(
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
            judge_score_max=params.judge_score_max,
            store_context_max_chars=params.store_context_max_chars,
            shared_curated_ids=shared_curated_ids,
            dataset_names=dataset_names,
        )
        summary = summarize(
            rows,
            params=params,
            distinct_ask_count=distinct_ask_count,
            sink_cluster_sizes=sink_cluster_sizes,
        )
        counters = run_counters(
            rows,
            recall_row_count=recall_row_count,
            distinct_ask_count=distinct_ask_count,
            collapsed_retry_count=collapsed_retry_count,
            taxonomy_version=taxonomy_version,
        )

        return await persist_run_results(run_id, rows, summary, counters, params=params)
    except Exception as error:
        logger.error("recall_coverage: run %s failed: %s", run_id, error, exc_info=True)
        # Bounded and class-prefixed, like the per-row errors: this string is
        # returned by the API as RunInfo.error. The log line above keeps the
        # full traceback.
        await fail_run(run_id, error_text(error))
        raise


async def _persist_empty(
    run_id: UUID,
    *,
    params: CoverageParams,
    owner_id: UUID,
    recall_row_count: int,
    distinct_ask_count: int,
    collapsed_retry_count: int,
) -> RunRecord:
    """Complete a run that had nothing to judge, counters intact."""
    summary = summarize([], params=params, distinct_ask_count=distinct_ask_count)
    counters = run_counters(
        [],
        recall_row_count=recall_row_count,
        distinct_ask_count=distinct_ask_count,
        collapsed_retry_count=collapsed_retry_count,
        taxonomy_version=await current_taxonomy_version(owner_id),
    )
    logger.debug("recall_coverage: run %s completed over an empty window", run_id)
    return await persist_run_results(run_id, [], summary, counters, params=params)


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
) -> list[int]:
    """Propose topics from dense sink clusters; return every sink cluster's size.

    The returned sizes are the sizes of **all** clusters at
    ``sink_cluster_threshold``, not of the suggestions that were written. The
    ``large_sink_cluster`` alert is a statement about the shape of the sink — a
    dense unmatched theme exists whether or not the owner has already dismissed a
    suggestion for it, and whether or not its label generation succeeded — so
    deriving the alert from the surviving suggestions would silence it exactly
    when it is most warranted.
    """
    if not sink_indices:
        return []

    sink_texts = [questions[index].text for index in sink_indices]
    sink_vectors = question_vectors[list(sink_indices)]

    groups, _comparisons = group_by_similarity(sink_vectors, params.sink_cluster_threshold)
    sizes = [len(group) for group in groups]

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
        "recall_coverage: %s sink questions produced %s suggestions",
        len(sink_indices),
        len(suggestions),
    )
    return sizes


__all__ = [
    "build_params",
    "run_recall_coverage",
    "schedule_recall_coverage_run",
    "start_recall_coverage_run",
]
