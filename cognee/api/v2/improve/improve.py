from uuid import UUID
from typing import Union, Optional, List, Type, Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict

from cognee.shared.logging_utils import get_logger

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
    from cognee.modules.users.methods import get_default_user

    user = kwargs.pop("user", None)
    if user is None:
        user = await get_default_user()

    feedback_alpha = kwargs.pop("feedback_alpha", 0.1)

    # Stage 1 & 2: bridge sessions into the permanent graph
    if session_ids:
        await _bridge_sessions(
            dataset=dataset,
            session_ids=session_ids,
            user=user,
            feedback_alpha=feedback_alpha,
            run_in_background=run_in_background,
        )

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

    # Stage 4: sync enriched graph back to session cache (incremental)
    if session_ids:
        await _sync_graph_to_sessions(
            dataset=dataset,
            session_ids=session_ids,
            user=user,
        )

    return result


async def _bridge_sessions(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
    feedback_alpha: float,
    run_in_background: bool,
):
    """Run feedback weights and session persistence pipelines."""

    # Stage 1: apply feedback weights from session scores
    from cognee.memify_pipelines.apply_feedback_weights import apply_feedback_weights_pipeline

    dataset_name = dataset if isinstance(dataset, str) else "main_dataset"

    try:
        await apply_feedback_weights_pipeline(
            user=user,
            session_ids=session_ids,
            dataset=dataset_name,
            alpha=feedback_alpha,
            run_in_background=False,
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
            run_in_background=False,
        )
        logger.info("improve: session Q&A persisted from %d session(s)", len(session_ids))
    except Exception as e:
        logger.warning("improve: session persistence failed (non-fatal): %s", e)


async def _sync_graph_to_sessions(
    dataset: Union[str, UUID],
    session_ids: List[str],
    user,
):
    """Incrementally sync recent graph knowledge into each session cache."""
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.tasks.memify.sync_graph_to_session import sync_graph_to_session

    dataset_name = dataset if isinstance(dataset, str) else "main_dataset"

    try:
        _, authorized_datasets = await resolve_authorized_user_datasets(dataset, user)
        if not authorized_datasets:
            logger.warning("improve: no authorized datasets for graph sync")
            return
        dataset_obj = authorized_datasets[0]
        user_id = str(user.id) if hasattr(user, "id") else None
        if not user_id:
            return

        for session_id in session_ids:
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
        logger.warning("improve: graph-to-session sync failed (non-fatal): %s", e)
