"""Build centroid-slot truth coordinates for a dataset's session learnings.

The ``session_learnings`` node set is replayed on every build into up to
``DEFAULT_K`` deterministic centroid slots. DocumentChunk nodes are projected
onto those slots and persisted with the centroid epoch used to compute them.
"""

from datetime import datetime, timezone
from typing import List, Optional, Union
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
    get_embedding_engine,
)
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.models import NodeSet
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


async def _embed_in_batches(embedding_engine, texts: List[str]) -> List[List[float]]:
    """Embed ``texts`` in bounded batches, preserving order. Fail-open -> []."""
    vectors: List[List[float]] = []
    for start in range(0, len(texts), NODE_EMBED_BATCH_SIZE):
        batch = texts[start : start + NODE_EMBED_BATCH_SIZE]
        try:
            vectors.extend(await embedding_engine.embed_text(batch))
        except Exception as error:
            logger.warning("truth_subspace: node embedding batch failed open: %s", error)
            # Keep alignment: pad failed batch with empty vectors (NEUTRAL coords).
            vectors.extend([[] for _ in batch])
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
    """Build/refresh centroid slots and chunk coordinates for ``dataset``."""
    resolved_user = user if user is not None else session_user.get()
    if resolved_user is None or getattr(resolved_user, "id", None) is None:
        resolved_user = await get_default_user()

    empty_result = {"anchors": 0, "nodes_scored": 0, "signature": "", "truth_epoch": 0}

    dataset_obj = await _resolve_dataset(dataset, resolved_user)
    if dataset_obj is None:
        logger.warning("truth_subspace: dataset %s not found or not writable", dataset)
        return empty_result

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()

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

        embedding_engine = get_embedding_engine()
        try:
            learning_vecs = await embedding_engine.embed_text(learning_texts)
        except Exception as error:
            logger.warning("truth_subspace: learning embedding failed open: %s", error)
            return {
                "anchors": len(existing_centroids),
                "nodes_scored": 0,
                "signature": signature,
                "truth_epoch": previous_epoch,
            }

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

        if centroids_changed(existing_centroids, rebuilt_centroids):
            current_epoch = previous_epoch + 1
            centroids = build_for_epoch(current_epoch)
            try:
                await upsert_centroids(vector_engine, centroids)
            except Exception as error:
                logger.warning("truth_subspace: centroid upsert failed open: %s", error)
                return {
                    "anchors": len(centroids),
                    "nodes_scored": 0,
                    "signature": signature,
                    "truth_epoch": current_epoch,
                }
        else:
            current_epoch = previous_epoch
            centroids = existing_centroids

        centroid_vecs = [centroid.centroid for centroid in centroids]

        # Step 5: LOAD nodes — ALL DocumentChunk nodes in the dataset (the chunk
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
            return {
                "anchors": len(centroids),
                "nodes_scored": 0,
                "signature": signature,
                "truth_epoch": current_epoch,
            }

        chunk_label = DocumentChunk.__name__
        scored: dict = {}
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
            logger.info("truth_subspace: %d centroids, no scoreable nodes", len(centroids))
            return {
                "anchors": len(centroids),
                "nodes_scored": 0,
                "signature": signature,
                "truth_epoch": current_epoch,
            }

        # Step 6: EMBED node texts (batched) and compute coords per node.
        node_vecs = await _embed_in_batches(embedding_engine, node_texts)
        for node_id, node_vec in zip(node_ids, node_vecs):
            try:
                coords = pad_coords(align.node_coords(node_vec, centroid_vecs), k)
                scored[node_id] = {
                    "truth_alignment": coords,
                    "truth_epoch": current_epoch,
                }
            except Exception as error:
                # Per-node fail-open: one bad node never sinks the batch.
                logger.debug("truth_subspace: coords failed for node %s: %s", node_id, error)

        if not scored:
            return {
                "anchors": len(centroids),
                "nodes_scored": 0,
                "signature": signature,
                "truth_epoch": current_epoch,
            }

        # Step 7: PERSIST per-node coordinate vectors.
        try:
            write_result = await graph_engine.set_node_truth_state(scored)
        except Exception as error:
            logger.warning("truth_subspace: persisting alignments failed open: %s", error)
            return {
                "anchors": len(centroids),
                "nodes_scored": 0,
                "signature": signature,
                "truth_epoch": current_epoch,
            }

        nodes_scored = sum(1 for ok in write_result.values() if ok)

    logger.info(
        "truth_subspace: built subspace -> centroids=%d nodes_scored=%d epoch=%d signature=%s",
        len(centroids),
        nodes_scored,
        current_epoch,
        signature,
    )
    return {
        "anchors": len(centroids),
        "nodes_scored": nodes_scored,
        "signature": signature,
        "truth_epoch": current_epoch,
    }
