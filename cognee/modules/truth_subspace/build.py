"""Build centroid-slot truth coordinates for a dataset's session learnings.

The ``session_learnings`` node set is replayed on every build into up to
``DEFAULT_K`` deterministic centroid slots. DocumentChunk nodes are projected
onto those slots and persisted with the centroid epoch used to compute them.

Write order (plan item B4). The hybrid reranker only trusts a chunk's
coordinates when the chunk's ``truth_epoch`` equals the live centroids' epoch,
so the centroids are the commit point. A build that moves the epoch from N to
N+1 therefore scores and persists the chunk coordinates at N+1 *first* and
upserts the N+1 centroids *last*: a failure anywhere before that final write
leaves epoch N live and every chunk it had scored still valid. Chunks whose
embedding batch failed are skipped outright rather than stored as all-zero
"neutral" coordinates, and the backend's ``set_node_truth_state`` support is
probed before the first embedding call.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.models import NodeSet
from cognee.modules.improve.capabilities import probe_graph_capabilities
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

from . import align
from .centroids import (
    build_centroids_from_learning_vectors,
    centroids_changed,
    extend_centroids_with_learning_vectors,
    learning_id,
    load_centroids,
    pad_coords,
    upsert_centroids,
)
from .constants import DEFAULT_K, TRUTH_NODE_SET, truth_session_node_set

logger = get_logger("truth_subspace")

# Node text embedding batch size — keep memory bounded on large subgraphs.
NODE_EMBED_BATCH_SIZE = 64

# Result statuses: PipelineRunInfo's vocabulary plus "skipped" (plan Part 5.4).
STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_ERRORED = "errored"

# Skip reason when the graph adapter has no ``set_node_truth_state``.
REASON_BACKEND_UNSUPPORTED = "backend_unsupported"


def _result(
    *,
    anchors: int,
    nodes_scored: int,
    signature: str,
    truth_epoch: int,
    status: str = STATUS_COMPLETED,
    nodes_skipped: int = 0,
    reason: Optional[str] = None,
    error: Optional[str] = None,
) -> dict:
    """The build's return shape. Integer fields double as stage counts."""
    result: Dict[str, Any] = {
        "anchors": int(anchors),
        "nodes_scored": int(nodes_scored),
        "nodes_skipped": int(nodes_skipped),
        "signature": signature,
        "truth_epoch": int(truth_epoch),
        "status": status,
    }
    if reason is not None:
        result["reason"] = reason
    if error is not None:
        result["error"] = error
    return result


def _node_index_text(node_data: dict) -> str:
    """Extract a node's index text for embedding (DocumentChunk -> ``text``)."""
    if not isinstance(node_data, dict):
        return ""
    text = node_data.get("text") or node_data.get("name") or ""
    return str(text).strip()


def _truth_node_sets(session_ids: Optional[List[str]]) -> List[str]:
    if not session_ids:
        return TRUTH_NODE_SET
    return [truth_session_node_set(session_id) for session_id in session_ids if session_id]


async def _fetch_learning_statements(graph_engine, session_ids: Optional[List[str]]) -> List[str]:
    """Read accepted lesson statements from the session_learnings node set.

    Traverses the ``session_learnings`` NodeSet to its member DocumentChunk
    nodes and returns their de-duplicated text. This is query-free: a vector
    search would require a query vector, but here we just want every lesson in
    the set. Fail-open -> [].
    """
    try:
        nodes, _edges = await graph_engine.get_nodeset_subgraph(
            node_type=NodeSet,
            node_name=_truth_node_sets(session_ids),
        )
    except Exception as error:
        logger.warning("truth_subspace: learning lookup failed open: %s", error)
        return []

    statements: List[str] = []
    seen = set()
    for _node_id, node_data in nodes or []:
        if not isinstance(node_data, dict):
            continue
        if node_data.get("type") != DocumentChunk.__name__:
            continue
        text = str(node_data.get("text") or "").strip()
        key = text.casefold()
        if text and key not in seen:
            statements.append(text)
            seen.add(key)
    return statements


async def _embed_in_batches(embedding_engine, texts: List[str]) -> List[Optional[List[float]]]:
    """Embed ``texts`` in bounded batches, preserving order.

    A failed batch yields ``None`` for each of its texts so the caller can skip
    those nodes. It never yields a placeholder vector: an empty or all-zero
    vector would project to all-zero "neutral" coordinates and be persisted as
    if it were a real score.
    """
    vectors: List[Optional[List[float]]] = []
    for start in range(0, len(texts), NODE_EMBED_BATCH_SIZE):
        batch = texts[start : start + NODE_EMBED_BATCH_SIZE]
        try:
            batch_vectors = list(await embedding_engine.embed_text(batch))
        except Exception as error:
            logger.warning(
                "truth_subspace: node embedding batch of %d failed, skipping those nodes: %s",
                len(batch),
                error,
            )
            vectors.extend([None for _ in batch])
            continue
        if len(batch_vectors) != len(batch):
            logger.warning(
                "truth_subspace: node embedding batch returned %d vectors for %d texts, "
                "skipping those nodes",
                len(batch_vectors),
                len(batch),
            )
            vectors.extend([None for _ in batch])
            continue
        vectors.extend(batch_vectors)
    return vectors


async def _resolve_dataset(dataset: Union[str, UUID], user):
    """Resolve a writable dataset object for ``user`` (or None)."""
    datasets = await get_authorized_existing_datasets([dataset], "write", user)
    return datasets[0] if datasets else None


async def build_truth_subspace(
    dataset: Union[str, UUID],
    session_ids: Optional[List[str]],
    user=None,
    k: int = DEFAULT_K,
) -> dict:
    """Build/refresh centroid slots and chunk coordinates for ``dataset``.

    Returns a dict with ``anchors``, ``nodes_scored``, ``nodes_skipped``,
    ``signature``, ``truth_epoch`` and ``status`` (``completed`` / ``skipped`` /
    ``errored``), plus ``reason`` when skipped and ``error`` when errored.
    ``truth_epoch`` is always the epoch that is *live* after the call: a build
    that failed before its final centroid write reports the previous epoch.
    """
    resolved_user = user if user is not None else session_user.get()
    if resolved_user is None or getattr(resolved_user, "id", None) is None:
        resolved_user = await get_default_user()

    empty_result = _result(anchors=0, nodes_scored=0, signature="", truth_epoch=0)

    dataset_obj = await _resolve_dataset(dataset, resolved_user)
    if dataset_obj is None:
        logger.warning("truth_subspace: dataset %s not found or not writable", dataset)
        return empty_result

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = await get_vector_engine_async()
        graph_engine = await get_graph_engine()

        # Step 0: capability gate, before any embedding call. Only Ladybug
        # implements set_node_truth_state today; on any other adapter the
        # build would embed every chunk and then fail at the final write.
        capabilities = probe_graph_capabilities(graph_engine)
        if not capabilities.supports_truth_state:
            logger.info(
                "truth_subspace: graph adapter %s has no truth-state support, skipping",
                capabilities.adapter,
            )
            return _result(
                anchors=0,
                nodes_scored=0,
                signature="",
                truth_epoch=0,
                status=STATUS_SKIPPED,
                reason=REASON_BACKEND_UNSUPPORTED,
            )

        # Step 1: accepted learning statements from session_learnings.
        statements = await _fetch_learning_statements(graph_engine, session_ids)
        if not statements:
            logger.info("truth_subspace: no learnings found, nothing to build")
            return empty_result

        try:
            existing_centroids = await load_centroids(vector_engine, str(dataset_obj.id), k)
        except Exception as error:
            logger.debug("truth_subspace: centroid load failed open: %s", error)
            existing_centroids = []

        previous_epoch = max((centroid.truth_epoch for centroid in existing_centroids), default=0)
        learning_items = sorted(
            {
                learning_id(statement): statement
                for statement in statements
                if str(statement).strip()
            }.items(),
            key=lambda item: item[0],
        )
        learning_ids = [item[0] for item in learning_items]
        learning_texts = [item[1] for item in learning_items]
        signature = align.stable_signature(learning_ids)

        def live_epoch_result(
            *, nodes_scored: int = 0, nodes_skipped: int = 0, error: Optional[str] = None
        ) -> dict:
            """Report the state that is live: epoch N's centroids, untouched."""
            return _result(
                anchors=len(existing_centroids),
                nodes_scored=nodes_scored,
                nodes_skipped=nodes_skipped,
                signature=signature,
                truth_epoch=previous_epoch,
                status=STATUS_ERRORED if error else STATUS_COMPLETED,
                error=error,
            )

        embedding_engine = get_embedding_engine()
        try:
            learning_vecs = await embedding_engine.embed_text(learning_texts)
        except Exception as error:
            logger.warning("truth_subspace: learning embedding failed open: %s", error)
            return live_epoch_result(error=f"learning embedding failed: {error}")

        updated_at = int(datetime.now(timezone.utc).timestamp() * 1000)
        learning_vectors = list(zip(learning_ids, learning_vecs))

        def build_for_epoch(truth_epoch: int):
            if session_ids:
                return extend_centroids_with_learning_vectors(
                    str(dataset_obj.id),
                    existing_centroids,
                    learning_vectors,
                    truth_epoch=truth_epoch,
                    updated_at=updated_at,
                    k=k,
                )
            return build_centroids_from_learning_vectors(
                str(dataset_obj.id),
                learning_vectors,
                truth_epoch=truth_epoch,
                updated_at=updated_at,
                k=k,
            )

        rebuilt_centroids = build_for_epoch(previous_epoch)
        if not rebuilt_centroids:
            return empty_result

        # Step 2: decide the epoch. New centroids are held in memory until the
        # chunks have been scored against them; nothing is written yet.
        if centroids_changed(existing_centroids, rebuilt_centroids):
            current_epoch = previous_epoch + 1
            centroids = build_for_epoch(current_epoch)
            centroids_pending_write = True
        else:
            current_epoch = previous_epoch
            centroids = existing_centroids
            centroids_pending_write = False

        centroid_vecs = [centroid.centroid for centroid in centroids]

        async def commit_centroids() -> Optional[str]:
            """Write the N+1 centroids; the last step of an epoch move. Returns an error."""
            if not centroids_pending_write:
                return None
            try:
                await upsert_centroids(vector_engine, centroids)
            except Exception as error:
                logger.warning("truth_subspace: centroid upsert failed open: %s", error)
                return f"centroid upsert failed: {error}"
            return None

        # Step 3: LOAD nodes — ALL DocumentChunk nodes in the dataset (the chunk
        # lane the hybrid retriever reranks). Scoping to the session_learnings
        # node set would only score the lessons themselves, never the corpus
        # chunks a query actually retrieves, so reranking would be a no-op.
        #
        # Use get_graph_data (sequential queries) and filter by type in memory.
        # get_filtered_graph_data runs its node/edge queries via asyncio.gather,
        # which deadlocks on the single-connection Kuzu subprocess backend.
        try:
            nodes, _edges = await graph_engine.get_graph_data()
        except Exception as error:
            logger.warning("truth_subspace: node load failed open: %s", error)
            return live_epoch_result(error=f"node load failed: {error}")

        chunk_label = DocumentChunk.__name__
        node_ids: List[str] = []
        node_texts: List[str] = []
        for node_id, node_data in nodes:
            if not isinstance(node_data, dict) or node_data.get("type") != chunk_label:
                continue
            text = _node_index_text(node_data)
            if not node_id or not text:
                continue
            node_ids.append(str(node_id))
            node_texts.append(text)

        if not node_texts:
            # Nothing to score, so nothing can be left inconsistent: the new
            # centroids can go live on their own.
            logger.info("truth_subspace: %d centroids, no scoreable nodes", len(centroids))
            commit_error = await commit_centroids()
            if commit_error:
                return live_epoch_result(error=commit_error)
            return _result(
                anchors=len(centroids),
                nodes_scored=0,
                signature=signature,
                truth_epoch=current_epoch,
            )

        # Step 4: EMBED node texts (batched) and compute coords per node at the
        # target epoch. Nodes whose batch failed are skipped, never zero-filled.
        node_vecs = await _embed_in_batches(embedding_engine, node_texts)
        scored: dict = {}
        nodes_skipped = 0
        for node_id, node_vec in zip(node_ids, node_vecs):
            if node_vec is None:
                nodes_skipped += 1
                continue
            try:
                coords = pad_coords(align.node_coords(node_vec, centroid_vecs), k)
                scored[node_id] = {
                    "truth_alignment": coords,
                    "truth_epoch": current_epoch,
                }
            except Exception as error:
                # Per-node fail-open: one bad node never sinks the batch.
                logger.debug("truth_subspace: coords failed for node %s: %s", node_id, error)
                nodes_skipped += 1

        if not scored:
            # Every chunk failed to embed. Moving the epoch now would leave the
            # whole corpus stale against the live centroids, so keep epoch N.
            logger.warning(
                "truth_subspace: no node could be scored (%d skipped), keeping epoch %d live",
                nodes_skipped,
                previous_epoch,
            )
            return live_epoch_result(
                nodes_skipped=nodes_skipped, error="no node embeddings available"
            )

        # Step 5: PERSIST per-node coordinate vectors at epoch N+1 first ...
        try:
            write_result = await graph_engine.set_node_truth_state(scored)
        except Exception as error:
            logger.warning("truth_subspace: persisting alignments failed open: %s", error)
            return live_epoch_result(
                nodes_skipped=nodes_skipped, error=f"persisting alignments failed: {error}"
            )

        nodes_scored = sum(1 for ok in write_result.values() if ok)

        # Step 6: ... and only then move the centroids to N+1. Until this write
        # lands, epoch N is what the reranker sees.
        commit_error = await commit_centroids()
        if commit_error:
            return live_epoch_result(
                nodes_scored=nodes_scored, nodes_skipped=nodes_skipped, error=commit_error
            )

    logger.info(
        "truth_subspace: built subspace -> centroids=%d nodes_scored=%d nodes_skipped=%d "
        "epoch=%d signature=%s",
        len(centroids),
        nodes_scored,
        nodes_skipped,
        current_epoch,
        signature,
    )
    return _result(
        anchors=len(centroids),
        nodes_scored=nodes_scored,
        nodes_skipped=nodes_skipped,
        signature=signature,
        truth_epoch=current_epoch,
    )
