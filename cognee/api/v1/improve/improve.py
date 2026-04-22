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
    COGNEE_GRAPH_EDGES_SYNCED,
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

    3. **Default enrichment** -- triplet embeddings are extracted and
       indexed (same as calling ``improve()`` without sessions).

    4. **Sync graph to session cache** -- incrementally copies new graph
       relationships back into the session cache as human-readable
       summaries for fast retrieval during session completions.

    Without ``session_ids``, only stage 3 runs.

    Args:
        dataset: Dataset name or UUID to process.
        run_in_background: Run processing asynchronously.
        node_name: Filter graph to specific named entities.
        session_ids: Session IDs whose feedback and Q&A content
            should be bridged into the permanent graph.
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
        else:
            release_improve_lock = None  # type: ignore[assignment]

        # Stage 1 & 2: bridge sessions into the permanent graph
        if session_ids:
            try:
                await _bridge_sessions(
                    dataset=dataset,
                    session_ids=session_ids,
                    user=user,
                    feedback_alpha=feedback_alpha,
                    run_in_background=run_in_background,
                )
                stages_run.extend(["feedback_weights", "persist_sessions"])

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
            except Exception:
                if acquired_lock_for:
                    from cognee.infrastructure.locks import release_improve_lock

                    await release_improve_lock(acquired_lock_for)
                raise

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

        # Stage 4: sync enriched graph back to session cache (incremental)
        # Skip when running in background — stage 3 hasn't completed yet
        if session_ids and not run_in_background:
            await _sync_graph_to_sessions(
                dataset=dataset,
                session_ids=session_ids,
                user=user,
            )
            stages_run.append("sync_graph_to_sessions")

        span.set_attribute(COGNEE_IMPROVE_STAGES, ",".join(stages_run))

        if acquired_lock_for:
            from cognee.infrastructure.locks import release_improve_lock

            await release_improve_lock(acquired_lock_for)

        return result


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
    """Run feedback weights and session persistence pipelines.

    Stage 1 (feedback weights): Updates ``feedback_weight`` on graph nodes
    and edges that were *used during retrieval* in session Q&A entries.
    Only elements referenced in ``used_graph_element_ids`` are affected.
    If no retrieval has occurred in these sessions, no weights are updated.

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


async def _sync_graph_to_sessions(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
):
    """Incrementally sync recent graph knowledge into each session cache.

    Reads new edges from the relational DB (since last checkpoint) and
    stores them as structured JSON-lines in the session's graph knowledge
    context. Each session is synced independently — one failure does not
    prevent others from completing.
    """
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.tasks.memify.sync_graph_to_session import sync_graph_to_session

    dataset_name = await _resolve_dataset_name(dataset, user)

    try:
        _, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
    except Exception as e:
        logger.warning("improve: graph-to-session sync setup failed (non-fatal): %s", e)
        return

    if not authorized_datasets:
        logger.warning("improve: no authorized datasets for graph sync")
        return
    dataset_obj = authorized_datasets[0]
    user_id = str(user.id) if hasattr(user, "id") else None
    if not user_id:
        return

    for session_id in session_ids:
        try:
            result = await sync_graph_to_session(
                user_id=user_id,
                session_id=session_id,
                dataset_id=dataset_obj.id,
                dataset_name=dataset_name,
            )
            logger.info(
                "improve: synced %d edges to session '%s'",
                result.get("synced", 0),
                session_id,
            )
        except Exception as e:
            logger.warning(
                "improve: graph-to-session sync failed for session '%s' (non-fatal): %s",
                session_id,
                e,
            )
