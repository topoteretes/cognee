import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.infrastructure.locks import ImproveLockRelease
from cognee.modules.observability import (
    COGNEE_DATASET_NAME,
    COGNEE_IMPROVE_STAGES,
    COGNEE_SESSION_ID,
    new_span,
)
from cognee.modules.operations import record_operation
from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
    resolve_authorized_user_datasets,
)
from cognee.modules.session_bridge import SessionPendingWork, probe_sessions_pending_work
from cognee.shared.logging_utils import get_logger

logger = get_logger("improve")

# Upper bound on "one more pass" reruns a single lock hold performs when other
# callers keep finding the lock busy. Whatever is still above the watermarks
# after that is picked up by the next trigger.
IMPROVE_MAX_RERUN_PASSES = 3

# Strong refs for background improve tasks: the event loop keeps only weak
# references, so an un-anchored task can be garbage-collected mid-run. Tasks
# discard themselves on completion. The set is not capped: single-session
# improves are already bounded to one in-flight task per session by the
# improve lock (the only path the plugin uses), and multi-session improves
# serialize on the dataset lock inside memify. Unbounded but self-draining,
# the same trade-off as ``_BACKGROUND_PIPELINE_TASKS`` and
# ``_BACKGROUND_REMEMBER_TASKS``.
_BACKGROUND_IMPROVE_TASKS: set[asyncio.Task] = set()


class ImproveKwargs(TypedDict, total=False):
    """Power-user overrides for improve(). Most users never need these."""

    extraction_tasks: list
    enrichment_tasks: list
    data: Any
    node_type: type
    user: object
    vector_db_config: dict
    graph_db_config: dict
    feedback_alpha: float


@dataclass
class _SessionImprovePlan:
    """Everything one session-bridging improve run needs, resolved up front."""

    dataset: str | UUID
    write_dataset_ref: Any  # the resolved dataset UUID
    session_ids: list[str]
    user: Any
    feedback_alpha: float
    node_name: list[str] | None
    build_global_context_index: bool
    build_truth_subspace: bool
    forced: bool
    memify_kwargs: dict = field(default_factory=dict)
    # Set when this run holds the single-session improve lock.
    lock_session_id: str | None = None
    lock_token: str | None = None
    background: bool = False


def _status_payload(status: str, *, dataset_id: Any, session_ids: list[str], **extra: Any) -> dict:
    """The non-pipeline improve answers: ``busy``, ``no_op`` and ``accepted``."""
    return {
        "status": status,
        "dataset_id": str(dataset_id),
        "session_ids": list(session_ids),
        **extra,
    }


def _pending_stage_names(pending: dict[str, SessionPendingWork]) -> list[str]:
    names: set[str] = set()
    for work in pending.values():
        names.update(work.stages())
    return sorted(names)


async def improve(
    dataset: str | UUID = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: list[str] | None = None,
    session_ids: list[str] | None = None,
    build_global_context_index: bool = False,
    build_truth_subspace: bool = False,
    **kwargs: Unpack[ImproveKwargs],
):
    """Enrich an existing knowledge graph with additional context and rules.

    When ``session_ids`` is provided, the improvement pipeline runs four
    stages:

    1. **Apply feedback weights** -- session entries with feedback scores
       update ``feedback_weight`` on the graph nodes/edges that were used
       to produce those answers. Higher-rated answers boost their source
       nodes; lower-rated answers decrease them.

    2. **Persist session Q&A** -- the question/answer text from those
       sessions is cognified into the permanent graph, tagged with
       ``node_set="user_sessions_from_cache"``.

    2c. **Distill sessions** -- each session's gated active-guidance
       entries are curated into entity-anchored lessons and
       add+cognified into the graph (tagged ``session_learnings``).
       Sessions with no gated guidance produce nothing. This is what
       lets ``remember(session, self_improvement=True)`` cover session
       distillation without an explicit ``distill_session`` call.

    3. **Default enrichment** -- triplet embeddings are extracted and
       indexed (same as calling ``improve()`` without sessions).

    4. **Global context index** -- when ``build_global_context_index=True``,
       builds retrieval-ready bucket and root summaries over the graph's
       text summaries.

    Without ``session_ids``, only stage 3 runs by default.

    Every session stage is guarded by a per-session watermark or applied
    marker, so repeating an improve is safe. Since SDK-593 it is also cheap:
    before any stage starts, the session cache is probed for work above those
    watermarks. When nothing is pending the call returns
    ``{"status": "no_op", ...}`` at once — no LLM call, no pipeline run, no
    operation record. Stages with nothing to do are skipped individually.

    A single-session improve holds a cross-worker per-session lock (TTL
    ``IMPROVE_LOCK_TTL_SECONDS``) for the whole run. A caller that finds it
    held gets ``{"status": "busy", "holder_age_seconds": ..., "rerun_requested":
    True}`` immediately — again without an operation record — and the holder
    runs one more watermark-driven pass before releasing, so the caller's newer
    tail is covered without any retry.

    Args:
        dataset: Dataset name or UUID to process.
        run_in_background: Return as soon as the run is accepted. With
            ``session_ids`` the whole stage sequence (including agent-context
            extraction, distillation and enrichment) runs in one ordered
            background task that holds the lock until it finishes; the
            answer is ``{"status": "accepted", "background": True, ...}`` and
            the operation record is written when the run completes. Without
            ``session_ids`` the enrichment pipeline itself runs in the
            background, as before.
        node_name: Filter graph to specific named entities.
        session_ids: Session IDs whose feedback and Q&A content
            should be bridged into the permanent graph.
        build_global_context_index: Opt-in flag for building the global
            context index after default enrichment. Forces a run even when
            no session work is pending.
        build_truth_subspace: Opt-in flag (default ``False``) for building the
            truth subspace from distilled session learnings after distillation
            and before enrichment. Only runs when ``session_ids`` is provided.
            Forces a run even when no session work is pending.
        **kwargs: Additional options -- see ``ImproveKwargs``.

    Returns:
        Pipeline run info (same as ``cognee.memify()``) for a completed
        blocking run, or one of the status dicts described above
        (``status`` is ``"busy"``, ``"no_op"`` or ``"accepted"``).

    Example::

        # Enrich graph + bridge session feedback and content
        await cognee.improve(dataset="docs", session_ids=["chat_1", "chat_2"])

        # Enrich graph only (no session bridging)
        await cognee.improve(dataset="docs")
    """
    from cognee import __version__ as cognee_version
    from cognee.shared.utils import send_telemetry

    send_telemetry(
        "cognee.improve",
        kwargs.get("user", "sdk"),
        additional_properties={
            "dataset": str(dataset),
            "session_count": len(session_ids) if session_ids else 0,
            "session_ids": ",".join(session_ids) if session_ids else "",
            "run_in_background": run_in_background,
            "build_global_context_index": build_global_context_index,
            "build_truth_subspace": build_truth_subspace,
            "cognee_version": cognee_version,
        },
    )

    with new_span("cognee.api.improve") as span:
        span.set_attribute(COGNEE_DATASET_NAME, str(dataset))
        if session_ids:
            span.set_attribute(COGNEE_SESSION_ID, ",".join(session_ids))

        from cognee.api.v1.serve.state import get_remote_client

        client = get_remote_client()
        if client is not None:
            return await client.improve(dataset, node_name=node_name, **kwargs)

        from cognee.modules.users.methods import get_default_user

        user = kwargs.pop("user", None)
        if user is None:
            user = await get_default_user()

        # Preconditions run before the operation record is opened, so a busy or
        # no-op answer below writes no row. A precondition failure is still
        # recorded as a failed improve, exactly as before.
        try:
            # The pipeline-run log writers INSERT the operation-record columns
            # (user_id, outcome, tokens, ...), so an existing database must be
            # at the current Alembic head before the first write — same gate
            # as cognify().
            from cognee.modules.migrations.startup import run_migrations_and_block

            await run_migrations_and_block(dataset, user)

            # One write-level resolution, shared by every stage below — the same
            # resolver remember/memify use: names resolve
            # or are created for the caller; a missing or unauthorized UUID raises
            # DatasetNotFoundError instead of being silently retargeted. Downstream
            # always receives the resolved UUID, never a name: names are
            # owner-scoped, so a name collapsed from a *shared* dataset's UUID
            # would re-resolve to the caller's own same-named dataset inside the
            # pipelines.
            user, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
        except Exception:
            async with record_operation("improve", user=user):
                raise

        resolved_dataset = authorized_datasets[0]
        write_dataset_ref = resolved_dataset.id
        feedback_alpha = kwargs.pop("feedback_alpha", 0.1)
        memify_kwargs = _memify_kwargs(kwargs)
        forced = bool(
            build_global_context_index
            or build_truth_subspace
            or any(kwargs.get(key) for key in ("extraction_tasks", "enrichment_tasks", "data"))
        )

        if not session_ids:
            # Dataset-only enrichment: nothing to probe, no session lock.
            async with record_operation("improve") as operation_context:
                operation_context.set_user(user)
                operation_context.set_dataset(write_dataset_ref)
                stages_run: list[str] = []
                result = await _run_enrichment(
                    dataset=dataset,
                    user=user,
                    node_name=node_name,
                    run_in_background=run_in_background,
                    build_global_context_index=build_global_context_index,
                    memify_kwargs=memify_kwargs,
                    stages_run=stages_run,
                )
                span.set_attribute(COGNEE_IMPROVE_STAGES, ",".join(stages_run))
                return result

        plan = _SessionImprovePlan(
            dataset=dataset,
            write_dataset_ref=write_dataset_ref,
            session_ids=list(session_ids),
            user=user,
            feedback_alpha=feedback_alpha,
            node_name=node_name,
            build_global_context_index=build_global_context_index,
            build_truth_subspace=build_truth_subspace,
            forced=forced,
            memify_kwargs=memify_kwargs,
            background=run_in_background,
        )

        # Mutex: single-session improves serialize on the session's cross-worker
        # lock so auto-improve + idle-watcher + SessionEnd don't duplicate work.
        # Multi-session improves skip the lock — the pattern is rare and locking
        # N sessions at once is messy.
        if len(plan.session_ids) == 1:
            from cognee.infrastructure.locks import request_improve_rerun, try_acquire_improve_lock

            sole_session = plan.session_ids[0]
            token = await try_acquire_improve_lock(sole_session, user.id)
            if token is None:
                lock_status = await request_improve_rerun(sole_session, user.id)
                if not lock_status.busy:
                    # The holder released between our two calls; claim it now.
                    token = await try_acquire_improve_lock(sole_session, user.id)
                if token is None:
                    logger.info(
                        "improve: session '%s' already being improved (holder age %s s); "
                        "rerun requested, skipping",
                        sole_session,
                        lock_status.holder_age_seconds,
                    )
                    return _status_payload(
                        "busy",
                        dataset_id=write_dataset_ref,
                        session_ids=plan.session_ids,
                        session_id=sole_session,
                        holder_age_seconds=lock_status.holder_age_seconds,
                        rerun_requested=lock_status.rerun_requested,
                    )
            plan.lock_session_id = sole_session
            plan.lock_token = token

        handed_off = False
        try:
            pending = await _probe_pending(plan)
            if not plan.forced and not any(work.any for work in pending.values()):
                # Let go only if no busy caller asked for a pass in the meantime;
                # otherwise its newer tail is ours to cover and we fall through
                # into a real run.
                if await _release_plan_lock(plan) is not ImproveLockRelease.RERUN:
                    logger.info(
                        "improve: nothing above the watermarks for session(s) %s, no-op",
                        ",".join(plan.session_ids),
                    )
                    return _status_payload(
                        "no_op",
                        dataset_id=write_dataset_ref,
                        session_ids=plan.session_ids,
                        reason=_no_op_reason(),
                    )
                pending = await _probe_pending(plan)

            if run_in_background:
                task = asyncio.create_task(_run_session_improve_in_background(plan, pending))
                _BACKGROUND_IMPROVE_TASKS.add(task)
                task.add_done_callback(_BACKGROUND_IMPROVE_TASKS.discard)
                handed_off = True  # the task releases the lock when it is done
                return _status_payload(
                    "accepted",
                    dataset_id=write_dataset_ref,
                    session_ids=plan.session_ids,
                    background=True,
                    pending_stages=_pending_stage_names(pending),
                )

            result, stages_run = await _run_session_improve(plan, pending)
            span.set_attribute(COGNEE_IMPROVE_STAGES, ",".join(stages_run))
            return result
        finally:
            if plan.lock_token and not handed_off:
                await _release_plan_lock(plan, force=True)


def _memify_kwargs(kwargs: dict) -> dict:
    """Forward-only options for the stage-3 memify call."""
    memify_kwargs = dict(kwargs)
    if memify_kwargs.get("node_type") is None:
        from cognee.modules.engine.models.node_set import NodeSet

        memify_kwargs["node_type"] = NodeSet

    # The default memify tasks never read the projected graph: they
    # stream triplets straight from the graph DB (or no-op). Pass the
    # non-empty sentinel the other improve stages already use so
    # memify skips the full-graph projection. Custom tasks/data keep
    # the projection, since a caller-supplied task may consume it.
    if not any(memify_kwargs.get(key) for key in ("extraction_tasks", "enrichment_tasks", "data")):
        memify_kwargs["data"] = [{}]
    return memify_kwargs


def _no_op_reason() -> str:
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    if not get_session_manager().is_available:
        return "session_cache_unavailable"
    return "nothing_pending"


async def _probe_pending(plan: _SessionImprovePlan) -> dict[str, SessionPendingWork]:
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    session_manager = get_session_manager()
    if not session_manager.is_available:
        return {session_id: SessionPendingWork(session_id) for session_id in plan.session_ids}
    return await probe_sessions_pending_work(
        session_manager, user_id=str(plan.user.id), session_ids=plan.session_ids
    )


async def _release_plan_lock(
    plan: _SessionImprovePlan, *, force: bool = False
) -> ImproveLockRelease:
    """Release this run's lock.

    ``RERUN`` means a busy caller's request was pending: it is now consumed and
    we still hold the lock, so the caller must run another pass. ``RELEASED``
    and ``LOST`` both mean the lock is no longer ours; the plan forgets its
    token so a later call is a no-op.
    """
    if not plan.lock_token:
        return ImproveLockRelease.RELEASED

    from cognee.infrastructure.locks import release_improve_lock

    outcome = await release_improve_lock(
        plan.lock_session_id, plan.user.id, plan.lock_token, force=force
    )
    if outcome is not ImproveLockRelease.RERUN:
        plan.lock_token = None
    return outcome


async def _run_session_improve_in_background(
    plan: _SessionImprovePlan, pending: dict[str, SessionPendingWork]
) -> None:
    """Background driver: run the stages, honour rerun requests, release the lock.

    The task inherits the request's context (operation origin included), so the
    operation record it writes is attributed to the caller — ``background=True``
    on that record is what says the caller did not wait for it.
    """
    try:
        await _run_session_improve(plan, pending)
    except Exception as exc:
        logger.warning(
            "improve: background session improve failed for %s: %s",
            ",".join(plan.session_ids),
            exc,
            exc_info=True,
        )
    finally:
        if plan.lock_token:
            try:
                await _release_plan_lock(plan, force=True)
            except Exception:
                logger.warning(
                    "improve: failed to release the improve lock for session '%s'",
                    plan.lock_session_id,
                    exc_info=True,
                )


async def _run_session_improve(
    plan: _SessionImprovePlan, pending: dict[str, SessionPendingWork]
) -> tuple[Any, list[str]]:
    """One recorded improve operation over the session stages, with the rerun loop.

    Runs every stage that has pending work (or all of them when forced). While
    this run holds the session lock, a caller that found it busy may have
    requested one more pass. The release itself is what checks for that: it
    consumes a pending request instead of letting go (one atomic UPDATE each),
    so no request can land between "checked" and "released" and be lost. Each
    extra pass re-reads the watermarks and processes only the newer tail.
    Bounded by ``IMPROVE_MAX_RERUN_PASSES``; after that the lock is
    force-released and a pending flag is left for the next acquirer, which
    starts with a full pass anyway.
    """
    stages_run: list[str] = []
    result: Any = {}

    async with record_operation("improve") as operation_context:
        operation_context.set_user(plan.user)
        operation_context.set_dataset(plan.write_dataset_ref)
        if len(plan.session_ids) == 1:
            operation_context.set_session_id(plan.session_ids[0])
        operation_context.set_background(plan.background)

        passes = 0
        while True:
            passes += 1
            if plan.forced or any(work.any for work in pending.values()):
                result = await _run_session_stages(plan, pending, stages_run)

            if not plan.lock_token:
                break
            if passes >= IMPROVE_MAX_RERUN_PASSES:
                # Force-release without consuming: a still-pending request stays
                # on the row for the next acquirer, whose full pass covers it.
                logger.info(
                    "improve: session '%s' reached the %d-pass bound; any further "
                    "rerun request is left to the next trigger",
                    plan.lock_session_id,
                    passes,
                )
                await _release_plan_lock(plan, force=True)
                break
            if await _release_plan_lock(plan) is not ImproveLockRelease.RERUN:
                break
            pending = await _probe_pending(plan)
            logger.info(
                "improve: rerun requested for session '%s'; running pass %d over %s",
                plan.lock_session_id,
                passes + 1,
                ",".join(_pending_stage_names(pending)) or "nothing new",
            )
            stages_run.append("rerun")

    return result, stages_run


async def _run_session_stages(
    plan: _SessionImprovePlan,
    pending: dict[str, SessionPendingWork],
    stages_run: list[str],
) -> Any:
    """Run the session-bridging stages that have work, then enrichment."""

    def _sessions_with(flag: str) -> list[str]:
        if plan.forced:
            return list(plan.session_ids)
        return [
            session_id
            for session_id in plan.session_ids
            if getattr(pending.get(session_id, SessionPendingWork(session_id)), flag)
        ]

    # Stage 1 & 2: bridge sessions into the permanent graph
    feedback_sessions = _sessions_with("feedback_qas")
    persist_sessions = _sessions_with("new_qa")
    if feedback_sessions or persist_sessions:
        await _bridge_sessions(
            dataset=plan.write_dataset_ref,
            session_ids=plan.session_ids,
            user=plan.user,
            feedback_alpha=plan.feedback_alpha,
            run_in_background=False,
            feedback_session_ids=feedback_sessions,
            persist_session_ids=persist_sessions,
        )
        if feedback_sessions:
            stages_run.append("feedback_weights")
        if persist_sessions:
            stages_run.append("persist_sessions")

    # Stage 2b: persist agent trace steps (tool calls with per-step
    # feedback) into the graph. Without this, the plugin's trace activity
    # never reaches permanent memory — only QA entries do.
    trace_sessions = _sessions_with("new_traces_to_persist")
    if trace_sessions:
        await _persist_session_traces(
            dataset=plan.write_dataset_ref,
            session_ids=trace_sessions,
            user=plan.user,
            run_in_background=False,
        )
        stages_run.append("persist_trace_steps")

    # Stage 2b2: distill each session's agent traces into agent-profile
    # session-context lessons (the LLM batch pass) before distillation, so
    # those lessons are available as gated guidance for stage 2c.
    agent_context_sessions = _sessions_with("new_traces_for_agent_context")
    lessons_by_session: dict[str, int] = {}
    if agent_context_sessions:
        lessons_by_session = await _extract_agent_context_per_session(
            session_ids=agent_context_sessions, user=plan.user
        )
        if sum(lessons_by_session.values()):
            stages_run.append("extract_agent_context")

    # Stage 2c: distill each session's gated guidance into curated,
    # entity-anchored lessons and add+cognify them into the graph. A session
    # that just gained agent-context lessons is distilled too.
    distill_sessions = list(
        dict.fromkeys(
            _sessions_with("distillable_entries")
            + [session_id for session_id, count in lessons_by_session.items() if count]
        )
    )
    if distill_sessions:
        distilled = await _distill_sessions(
            dataset=plan.write_dataset_ref,
            session_ids=distill_sessions,
            user=plan.user,
        )
        if distilled:
            stages_run.append("distill_sessions")

    # Stage 2c2: fold rated turns and stated preferences into the
    # calling user's preference subgraph (weighted `prefers` edges
    # plus the preference node's text). One call for all sessions —
    # preferences aggregate across a user's sessions.
    preference_result = await _update_user_preferences(
        dataset=plan.write_dataset_ref,
        session_ids=plan.session_ids,
        user=plan.user,
    )
    if preference_result is not None and preference_result.status == "completed":
        stages_run.append("user_preferences")

    # Stage 2d: build the truth subspace from distilled session
    # learnings (opt-in, default OFF). Runs after distillation so
    # freshly accepted lessons are available as anchors, and before
    # enrichment. Non-fatal — never blocks the rest of improve().
    if plan.build_truth_subspace:
        try:
            from cognee.modules.truth_subspace.build import (
                build_truth_subspace as _build_truth_subspace,
            )

            result_ts = await _build_truth_subspace(
                dataset=plan.dataset,
                session_ids=plan.session_ids,
                user=plan.user,
            )
            logger.info("improve: truth subspace built -> %s", result_ts)
            stages_run.append("build_truth_subspace")
        except Exception as e:
            logger.warning(
                "improve: truth subspace build failed (non-fatal): %s",
                e,
                exc_info=True,
            )

    # Stage 3 (+4): default enrichment, then the optional global context index.
    # Ordered within this run, so the index sees the enrichment's output in
    # background mode too.
    return await _run_enrichment(
        dataset=plan.dataset,
        user=plan.user,
        node_name=plan.node_name,
        run_in_background=False,
        build_global_context_index=plan.build_global_context_index,
        memify_kwargs=plan.memify_kwargs,
        stages_run=stages_run,
    )


async def _run_enrichment(
    *,
    dataset: str | UUID,
    user,
    node_name: list[str] | None,
    run_in_background: bool,
    build_global_context_index: bool,
    memify_kwargs: dict,
    stages_run: list[str],
) -> Any:
    """Stage 3: default enrichment (triplet embeddings); stage 4: global context index."""
    from cognee.modules.memify import memify

    result = await memify(
        dataset=dataset,
        node_name=node_name,
        user=user,
        run_in_background=run_in_background,
        **memify_kwargs,
    )
    stages_run.append("memify_enrichment")

    if build_global_context_index:
        if run_in_background:
            logger.warning(
                "improve: global context index skipped in background mode "
                "because ordered background pipeline chaining is not supported"
            )
        else:
            global_context_index_updated = await _build_global_context_index(
                dataset=dataset,
                user=user,
            )
            if global_context_index_updated:
                stages_run.append("global_context_index")

    return result


async def _build_global_context_index(
    dataset: str | UUID,
    user,
) -> bool:
    from cognee.memify_pipelines.global_context_index import global_context_index_pipeline

    try:
        await global_context_index_pipeline(
            user=user,
            dataset=dataset,
            run_in_background=False,
            bucketing_strategy="graph",
            max_bucket_size=4,
        )
        logger.info("improve: global context index updated")
        return True
    except Exception as e:
        logger.warning(
            "improve: global context index update failed (non-fatal): %s", e, exc_info=True
        )
        return False


async def _bridge_sessions(
    dataset: str | UUID,
    session_ids: list[str],
    user,
    feedback_alpha: float,
    run_in_background: bool,
    feedback_session_ids: list[str] | None = None,
    persist_session_ids: list[str] | None = None,
):
    """Run feedback weights and session persistence pipelines.

    Stage 1 (feedback weights): Updates ``feedback_weight`` on graph nodes
    and edges that were *used during retrieval* in session Q&A entries.
    Only elements referenced in ``used_graph_element_ids`` are affected.
    If no retrieval has occurred in these sessions, no weights are updated.

    Stage 2 (persist Q&A): Cognifies the actual question/answer text from
    sessions into the permanent graph, tagged with
    ``node_set="user_sessions_from_cache"``. This persists the Q&A content
    itself, not serialized graph edges.

    ``feedback_session_ids`` / ``persist_session_ids`` narrow each stage to
    the sessions that have pending work; ``None`` means all of ``session_ids``.
    A stage whose list is empty is skipped.
    """
    feedback_session_ids = (
        list(session_ids) if feedback_session_ids is None else feedback_session_ids
    )
    persist_session_ids = list(session_ids) if persist_session_ids is None else persist_session_ids

    # Stage 1: apply feedback weights from session retrieval traces
    if feedback_session_ids:
        from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

        try:
            await apply_feedback_weights_pipeline(
                user=user,
                session_ids=feedback_session_ids,
                dataset=dataset,
                alpha=feedback_alpha,
                run_in_background=run_in_background,
            )
            logger.info(
                "improve: feedback weights applied from %d session(s)", len(feedback_session_ids)
            )
        except Exception as e:
            logger.warning("improve: feedback weights failed (non-fatal): %s", e, exc_info=True)

    # Stage 2: persist session Q&A into permanent graph
    if persist_session_ids:
        from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
            persist_sessions_in_knowledge_graph_pipeline,
        )

        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=persist_session_ids,
            dataset=dataset,
            run_in_background=run_in_background,
        )
        logger.info("improve: session Q&A persisted from %d session(s)", len(persist_session_ids))


async def _extract_agent_context_per_session(
    session_ids: list[str],
    user,
) -> dict[str, int]:
    """Flush pending trace windows into agent-profile lessons before distillation.

    Delegates to ``agent_context_extraction.extract_pending_agent_context`` per session, which
    shares the same watermark used by mid-session trace extraction. ``min_new_traces=1`` makes
    improve/session-end flush any remaining unprocessed traces before distillation. Gated on
    automatic session context and best-effort/fail-open: an error on one session never blocks the
    others or the rest of ``improve()``. Returns the number of lessons created/linked per session.
    """
    from cognee.infrastructure.session.agent_context_extraction import (
        extract_pending_agent_context,
    )
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    session_manager = get_session_manager()
    if not session_manager.is_available or not session_manager.is_auto_feedback_enabled():
        return {}

    user_id = str(user.id)
    touched: dict[str, int] = {}
    for session_id in session_ids:
        try:
            ids = await extract_pending_agent_context(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                min_new_traces=1,
            )
            touched[session_id] = len(ids)
        except Exception as e:
            logger.warning(
                "improve: agent-context extraction failed for '%s' (non-fatal): %s",
                session_id,
                e,
                exc_info=True,
            )
    return touched


async def _extract_agent_context(
    session_ids: list[str],
    user,
) -> int:
    """Total lessons created/linked across ``session_ids`` (see the per-session variant)."""
    return sum(
        (await _extract_agent_context_per_session(session_ids=session_ids, user=user)).values()
    )


async def _distill_sessions(
    dataset: str | UUID,
    session_ids: list[str],
    user,
) -> int:
    """Distill each session's gated learnings into curated lessons in the graph.

    Delegates to ``session_distillation.distill_session`` per session: it loads
    the session's gated active-guidance entries above the distillation
    watermark, curates them into proposed lessons, writes/rejects each with
    entity anchoring, and add+cognifies the accepted lessons into ``dataset``
    (tagged ``session_learnings``).

    Best-effort and fail-open: a session with no gated guidance simply yields no
    lessons (status ``no_gated_entries``), and an error on one session never
    blocks the others or the rest of ``improve()``. Returns the total number of
    lesson documents written across all sessions.

    Note: ``distill_session`` runs its own ``add``/``cognify`` (it does not call
    ``improve``), so there is no recursion back into this function.
    """
    from cognee.modules.session_distillation import distill_session

    distilled = 0
    for session_id in session_ids:
        try:
            result = await distill_session(session_id, dataset=dataset, user=user)
            distilled += len(result.documents)
            logger.info(
                "improve: distilled session '%s' -> status=%s documents=%d",
                session_id,
                result.status,
                len(result.documents),
            )
        except Exception as e:
            logger.warning(
                "improve: session distillation failed for '%s' (non-fatal): %s",
                session_id,
                e,
                exc_info=True,
            )
    return distilled


async def _update_user_preferences(
    dataset: str | UUID,
    session_ids: list[str],
    user,
):
    """Update the caller's per-dataset preference node and ``prefers`` weights.

    Delegates to ``user_preferences.update_user_preferences`` once for all
    sessions: rated turns move that user's ``prefers`` edge weights (exactly
    once per turn), idle edges decay against the per-turn clock and are pruned
    at neutral, and gated ``preferences`` session-context entries are folded
    into the preference node's text behind a watermark.

    Best-effort and fail-open like its neighbours: an error here never blocks
    the rest of ``improve()``. Returns the ``PreferenceUpdateResult`` (or None
    on error) so the caller can decide whether the stage actually changed
    anything.
    """
    try:
        from cognee.modules.user_preferences.update import update_user_preferences

        result = await update_user_preferences(
            session_ids=session_ids,
            dataset=dataset,
            user=user,
        )
        if result.status == "personalization_disabled":
            logger.debug("improve: user preference stage skipped (PERSONALIZATION_ENABLED is off)")
            return result
        logger.info(
            "improve: user preferences updated -> status=%s turns=%d edges=%d "
            "pruned=%d text_lines=%d",
            result.status,
            result.turns_applied,
            result.edges_written,
            result.edges_pruned,
            result.text_lines_added,
        )
        return result
    except Exception as e:
        logger.warning("improve: user preference update failed (non-fatal): %s", e, exc_info=True)
        return None


async def _persist_session_traces(
    dataset: str | UUID,
    session_ids: list[str],
    user,
    run_in_background: bool,
):
    """Cognify per-step agent trace feedbacks into the knowledge graph.

    Without this step, the Claude Code plugin's tool-call activity
    (the bulk of session data — hundreds of Bash/Edit/Read/Write trace
    steps per session) never makes it into permanent memory. Only QA
    entries do via ``persist_sessions_in_knowledge_graph_pipeline``.

    Runs the dedicated ``persist_agent_trace_feedbacks_in_knowledge_graph_pipeline``
    that extracts per-step ``session_feedback`` above the trace persist
    watermark from the cache and cognifies it into the ``agent_trace_feedbacks``
    node-set.
    """
    try:
        from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
            persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
        )

        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset,
            node_set_name="agent_trace_feedbacks",
            raw_trace_content=False,
            last_n_steps=None,  # persist all not-yet-persisted steps
            run_in_background=run_in_background,
        )
        logger.info(
            "improve: agent trace steps persisted from %d session(s)",
            len(session_ids),
        )
    except Exception as e:
        logger.warning("improve: trace persistence failed (non-fatal): %s", e, exc_info=True)
