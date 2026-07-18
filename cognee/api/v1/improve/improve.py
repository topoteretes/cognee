from uuid import UUID
from typing import Union, Optional, List, Type, Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.shared.logging_utils import get_logger
from cognee.modules.observability import (
    new_span,
    COGNEE_DATASET_NAME,
    COGNEE_SESSION_ID,
    COGNEE_IMPROVE_STAGES,
)

logger = get_logger("improve")


class ImproveKwargs(TypedDict, total=False):
    """Power-user overrides for improve(). Most users never need these."""

    extraction_tasks: list
    enrichment_tasks: list
    data: Any
    node_type: Type
    user: object
    vector_db_config: dict
    graph_db_config: dict
    feedback_alpha: float


async def improve(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: Optional[List[str]] = None,
    session_ids: Optional[List[str]] = None,
    build_global_context_index: bool = False,
    build_truth_subspace: bool = False,
    **kwargs: Unpack[ImproveKwargs],
):
    """Enrich an existing knowledge graph with additional context and rules.

    When ``session_ids`` is provided, the improvement pipeline runs session bridge
    stages:

    1. **Apply feedback weights** -- session entries with feedback scores
       update ``feedback_weight`` on the graph nodes/edges that were used
       to produce those answers. Higher-rated answers boost their source
       nodes; lower-rated answers decrease them.

    2. **Apply frequency weights** -- session entries with recorded graph
       element usage increment ``frequency_weight`` for the nodes/edges that
       were recalled, including unrated answers.

    3. **Persist session Q&A** -- the question/answer text from those
       sessions is cognified into the permanent graph, tagged with
       ``node_set="user_sessions_from_cache"``.

    3c. **Distill sessions** -- each session's gated active-guidance
       entries are curated into entity-anchored lessons and
       add+cognified into the graph (tagged ``session_learnings``).
       Sessions with no gated guidance produce nothing. This is what
       lets ``remember(session, self_improvement=True)`` cover session
       distillation without an explicit ``distill_session`` call.

    4. **Default enrichment** -- triplet embeddings are extracted and
       indexed (same as calling ``improve()`` without sessions).

    5. **Global context index** -- when ``build_global_context_index=True``,
       builds retrieval-ready bucket and root summaries over the graph's
       text summaries.

    Without ``session_ids``, only stage 3 runs by default.

    Args:
        dataset: Dataset name or UUID to process.
        run_in_background: Run processing asynchronously.
        node_name: Filter graph to specific named entities.
        session_ids: Session IDs whose feedback and Q&A content
            should be bridged into the permanent graph.
        build_global_context_index: Opt-in flag for building the global
            context index after default enrichment. Skipped in background
            mode because ordered background pipeline chaining is not
            supported yet.
        build_truth_subspace: Opt-in flag (default ``False``) for building the
            truth subspace from distilled session learnings after distillation
            and before enrichment. Only runs when ``session_ids`` is provided.
            Off by default = no behaviour change.
        **kwargs: Additional options -- see ``ImproveKwargs``.

    Returns:
        Pipeline run info (same as ``cognee.memify()``).

    Example::

        # Enrich graph + bridge session feedback and content
        await cognee.improve(dataset="docs", session_ids=["chat_1", "chat_2"])

        # Enrich graph only (no session bridging)
        await cognee.improve(dataset="docs")
    """
    from cognee.shared.utils import send_telemetry
    from cognee import __version__ as cognee_version

    stages_run = []

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

        feedback_alpha = kwargs.pop("feedback_alpha", 0.1)

        # Mutex: single-session improves serialize on the session's
        # lock so auto-improve + idle-watcher + SessionEnd don't
        # duplicate work. Multi-session improves skip the lock — the
        # pattern is rare and locking N sessions at once is messy.
        acquired_lock_for: Optional[str] = None
        if session_ids and len(session_ids) == 1:
            from cognee.infrastructure.locks import (
                release_improve_lock,
                try_acquire_improve_lock,
            )

            sole_session = session_ids[0]
            if not await try_acquire_improve_lock(sole_session):
                logger.info(
                    "improve: session '%s' already being improved, skipping",
                    sole_session,
                )
                return {}
            acquired_lock_for = sole_session

        try:
            # Stage 1 & 2: bridge sessions into the permanent graph
            if session_ids:
                await _bridge_sessions(
                    dataset=dataset,
                    session_ids=session_ids,
                    user=user,
                    feedback_alpha=feedback_alpha,
                    run_in_background=run_in_background,
                )
                stages_run.extend(["feedback_weights", "frequency_weights", "persist_sessions"])

                # Stage 2b: persist agent trace steps (tool calls with
                # per-step feedback) into the graph. Without this, the
                # plugin's trace activity never reaches permanent
                # memory — only QA entries do.
                await _persist_session_traces(
                    dataset=dataset,
                    session_ids=session_ids,
                    user=user,
                    run_in_background=run_in_background,
                )
                stages_run.append("persist_trace_steps")

                # Stage 2b2: distill each session's agent traces into agent-profile
                # session-context lessons (the LLM batch pass) before distillation, so
                # those lessons are available as gated guidance for stage 2c.
                if await _extract_agent_context(session_ids=session_ids, user=user):
                    stages_run.append("extract_agent_context")

                # Stage 2c: distill each session's gated guidance into curated,
                # entity-anchored lessons and add+cognify them into the graph.
                # This is what lets remember(session, self_improvement=True)
                # cover session distillation without an explicit
                # cognee.session.distill_session call.
                distilled = await _distill_sessions(
                    dataset=dataset,
                    session_ids=session_ids,
                    user=user,
                )
                if distilled:
                    stages_run.append("distill_sessions")

                # Stage 2d: build the truth subspace from distilled session
                # learnings (opt-in, default OFF). Runs after distillation so
                # freshly accepted lessons are available as anchors, and before
                # enrichment. Non-fatal — never blocks the rest of improve().
                if build_truth_subspace:
                    try:
                        from cognee.modules.truth_subspace.build import (
                            build_truth_subspace as _build_truth_subspace,
                        )

                        result_ts = await _build_truth_subspace(
                            dataset=dataset,
                            session_ids=session_ids,
                            user=user,
                        )
                        logger.info("improve: truth subspace built -> %s", result_ts)
                        stages_run.append("build_truth_subspace")
                    except Exception as e:
                        logger.warning("improve: truth subspace build failed (non-fatal): %s", e)

            # Stage 3: default enrichment (triplet embeddings)
            from cognee.modules.memify import memify

            if "node_type" not in kwargs or kwargs.get("node_type") is None:
                from cognee.modules.engine.models.node_set import NodeSet

                kwargs["node_type"] = NodeSet

            result = await memify(
                dataset=dataset,
                node_name=node_name,
                user=user,
                run_in_background=run_in_background,
                **kwargs,
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

            span.set_attribute(COGNEE_IMPROVE_STAGES, ",".join(stages_run))

            return result
        finally:
            if acquired_lock_for:
                from cognee.infrastructure.locks import release_improve_lock

                await release_improve_lock(acquired_lock_for)


async def _build_global_context_index(
    dataset: Union[str, UUID],
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
        logger.warning("improve: global context index update failed (non-fatal): %s", e)
        return False


async def _resolve_dataset_name(dataset: Union[str, UUID], user) -> str:
    """Resolve a dataset reference to its name string."""
    if isinstance(dataset, str):
        return dataset
    from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset

    ds = await get_authorized_dataset(user, dataset, "write")
    return ds.name if ds else "main_dataset"


async def _bridge_sessions(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
    feedback_alpha: float,
    run_in_background: bool,
):
    """Run feedback, frequency, and session persistence pipelines.

    Stage 1a (feedback weights): Updates ``feedback_weight`` on graph nodes
    and edges that were *used during retrieval* in session Q&A entries.
    Only elements referenced in ``used_graph_element_ids`` are affected.
    If no retrieval has occurred in these sessions, no weights are updated.

    Stage 1b (frequency weights): Increments ``frequency_weight`` on graph
    nodes and edges used during retrieval. Unlike feedback, this also applies
    to unrated QAs because recall frequency is a usage signal.

    Stage 2 (persist Q&A): Cognifies the actual question/answer text from
    sessions into the permanent graph, tagged with
    ``node_set="user_sessions_from_cache"``. This persists the Q&A content
    itself, not serialized graph edges.
    """

    # Stage 1: apply feedback weights from session retrieval traces
    from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

    dataset_name = await _resolve_dataset_name(dataset, user)

    try:
        await apply_feedback_weights_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset_name,
            alpha=feedback_alpha,
            run_in_background=run_in_background,
        )
        logger.info("improve: feedback weights applied from %d session(s)", len(session_ids))
    except Exception as e:
        logger.warning("improve: feedback weights failed (non-fatal): %s", e)

    try:
        from cognee.memify_pipelines.apply_frequency_weights import apply_frequency_weights_pipeline

        await apply_frequency_weights_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset_name,
            run_in_background=run_in_background,
        )
        logger.info("improve: frequency weights applied from %d session(s)", len(session_ids))
    except Exception as e:
        logger.warning("improve: frequency weights failed (non-fatal): %s", e)

    # Stage 2: persist session Q&A into permanent graph
    from cognee.memify_pipelines.persist_sessions_in_knowledge_graph import (
        persist_sessions_in_knowledge_graph_pipeline,
    )

    try:
        await persist_sessions_in_knowledge_graph_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset_name,
            run_in_background=run_in_background,
        )
        logger.info("improve: session Q&A persisted from %d session(s)", len(session_ids))
    except Exception as e:
        logger.warning("improve: session persistence failed (non-fatal): %s", e)


async def _extract_agent_context(
    session_ids: List[str],
    user,
) -> int:
    """Flush pending trace windows into agent-profile lessons before distillation.

    Delegates to ``agent_context_extraction.extract_pending_agent_context`` per session, which
    shares the same watermark used by mid-session trace extraction. ``min_new_traces=1`` makes
    improve/session-end flush any remaining unprocessed traces before distillation. Gated on
    automatic session context and best-effort/fail-open: an error on one session never blocks the
    others or the rest of ``improve()``. Returns the number of lessons created/linked.
    """
    from cognee.infrastructure.session.agent_context_extraction import (
        extract_pending_agent_context,
    )
    from cognee.infrastructure.session.get_session_manager import get_session_manager

    session_manager = get_session_manager()
    if not session_manager.is_available or not session_manager.is_auto_feedback_enabled():
        return 0

    user_id = str(user.id)
    touched = 0
    for session_id in session_ids:
        try:
            ids = await extract_pending_agent_context(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                min_new_traces=1,
            )
            touched += len(ids)
        except Exception as e:
            logger.warning(
                "improve: agent-context extraction failed for '%s' (non-fatal): %s",
                session_id,
                e,
            )
    return touched


async def _distill_sessions(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
) -> int:
    """Distill each session's gated learnings into curated lessons in the graph.

    Delegates to ``session_distillation.distill_session`` per session: it loads
    the session's gated active-guidance entries, curates them into proposed
    lessons, writes/rejects each with entity anchoring, and add+cognifies the
    accepted lessons into ``dataset`` (tagged ``session_learnings``).

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
            )
    return distilled


async def _persist_session_traces(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
    run_in_background: bool,
):
    """Cognify per-step agent trace feedbacks into the knowledge graph.

    Without this step, the Claude Code plugin's tool-call activity
    (the bulk of session data — hundreds of Bash/Edit/Read/Write trace
    steps per session) never makes it into permanent memory. Only QA
    entries do via ``persist_sessions_in_knowledge_graph_pipeline``.

    Runs the dedicated ``persist_agent_trace_feedbacks_in_knowledge_graph_pipeline``
    that extracts per-step ``session_feedback`` from the cache and
    cognifies it into the ``agent_trace_feedbacks`` node-set.
    """
    dataset_name = await _resolve_dataset_name(dataset, user)

    try:
        from cognee.memify_pipelines.persist_agent_trace_feedbacks_in_knowledge_graph import (
            persist_agent_trace_feedbacks_in_knowledge_graph_pipeline,
        )

        await persist_agent_trace_feedbacks_in_knowledge_graph_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset_name,
            node_set_name="agent_trace_feedbacks",
            raw_trace_content=False,
            last_n_steps=None,  # persist all stored steps on demand
            run_in_background=run_in_background,
        )
        logger.info(
            "improve: agent trace steps persisted from %d session(s)",
            len(session_ids),
        )
    except Exception as e:
        logger.warning("improve: trace persistence failed (non-fatal): %s", e)
