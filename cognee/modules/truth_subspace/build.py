"""Build the truth subspace for a dataset's distilled session learnings.

The truth subspace is a small set of accepted lesson statements (the
``session_learnings`` node set) used as semantic *anchors*. Each graph node is
projected onto those anchors and the resulting coordinate vector is persisted on
the node, so retrieval can later nudge scores toward statements the system has
already accepted as true.

The whole build is fail-open and OFF BY DEFAULT: callers opt in explicitly. When
there are no anchors, nothing is written and an empty-ish result is returned, so
baseline behaviour is untouched.

Flow:

1. ANCHORS   traverse the ``session_learnings`` node set -> anchor statements.
2. UPSERT    build :class:`TruthAnchor` data points and upsert them.
3. EMBED     embed the active anchor statements in memory.
4. ACTIVE K  deterministically pick the most-recent ``k`` anchors + signature.
5. LOAD      load the DocumentChunk subgraph and extract each node's index text.
6. COORDS    embed node texts (batched) and project onto the active anchors.
7. PERSIST   write per-node coordinate vectors via the graph engine.
"""

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
from .constants import DEFAULT_K, TRUTH_ANCHOR_COLLECTION, TRUTH_NODE_SET
from .models import TruthAnchor

logger = get_logger("truth_subspace")

# Node text embedding batch size — keep memory bounded on large subgraphs.
NODE_EMBED_BATCH_SIZE = 64


def _node_index_text(node_data: dict) -> str:
    """Extract a node's index text for embedding (DocumentChunk -> ``text``)."""
    if not isinstance(node_data, dict):
        return ""
    text = node_data.get("text") or node_data.get("name") or ""
    return str(text).strip()


async def _fetch_anchor_statements(graph_engine) -> List[str]:
    """Read accepted lesson statements from the session_learnings node set.

    Traverses the ``session_learnings`` NodeSet to its member DocumentChunk
    nodes and returns their de-duplicated text. This is query-free: a vector
    search would require a query vector, but here we just want every lesson in
    the set. Fail-open -> [].
    """
    try:
        nodes, _edges = await graph_engine.get_nodeset_subgraph(
            node_type=NodeSet,
            node_name=TRUTH_NODE_SET,
        )
    except Exception as error:
        logger.warning("truth_subspace: anchor lookup failed open: %s", error)
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
    """Build/refresh the truth subspace for ``dataset``.

    Reads accepted session-learning statements as anchors, upserts them as
    :class:`TruthAnchor` data points, embeds the active anchors, then projects
    every DocumentChunk node onto those anchors and persists the per-node
    coordinate vectors on the graph.

    Fail-open and logged throughout, mirroring the ``improve()`` distillation
    stage. Returns ``{"anchors": n, "nodes_scored": m, "signature": str}``.
    When there are no anchors, returns early writing nothing.
    """
    resolved_user = user if user is not None else session_user.get()
    if resolved_user is None or getattr(resolved_user, "id", None) is None:
        resolved_user = await get_default_user()

    empty_result = {"anchors": 0, "nodes_scored": 0, "signature": ""}

    dataset_obj = await _resolve_dataset(dataset, resolved_user)
    if dataset_obj is None:
        logger.warning("truth_subspace: dataset %s not found or not writable", dataset)
        return empty_result

    async with set_database_global_context_variables(dataset_obj.id, dataset_obj.owner_id):
        vector_engine = get_vector_engine()
        graph_engine = await get_graph_engine()

        # Step 1: ANCHORS — accepted lesson statements from session_learnings.
        statements = await _fetch_anchor_statements(graph_engine)
        if not statements:
            logger.info("truth_subspace: no anchors found, nothing to build")
            return empty_result

        # Step 2: UPSERT TruthAnchor data points. No belongs_to_set tag: the
        # query-time anchor search is unscoped (apply_node_filter=False), and a
        # NodeSet object does not serialize into the LanceDB anchor collection.
        anchors = [
            TruthAnchor(id=TruthAnchor.id_for(statement), statement=statement)
            for statement in statements
        ]
        try:
            # Pre-create the collection. create_data_points auto-creates a missing
            # collection while already holding VECTOR_DB_LOCK, then calls
            # create_collection which re-acquires the same non-reentrant lock —
            # a deadlock on first run. Creating it up front (lock acquired once)
            # makes create_data_points take its collection-exists fast path.
            await vector_engine.create_collection(
                TRUTH_ANCHOR_COLLECTION, payload_schema=TruthAnchor
            )
            await vector_engine.create_data_points(TRUTH_ANCHOR_COLLECTION, anchors)
        except Exception as error:
            logger.warning("truth_subspace: anchor upsert failed open: %s", error)

        # Step 3 & 4: ACTIVE K + signature, then embed the active anchors.
        active_anchors = align.active_anchor_order(anchors, k)
        active_statements = [anchor.statement for anchor in active_anchors]
        signature = align.anchor_signature([anchor.id for anchor in active_anchors])

        embedding_engine = get_embedding_engine()
        try:
            anchor_vecs = await embedding_engine.embed_text(active_statements)
        except Exception as error:
            logger.warning("truth_subspace: anchor embedding failed open: %s", error)
            return {"anchors": len(active_anchors), "nodes_scored": 0, "signature": signature}

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
            return {"anchors": len(active_anchors), "nodes_scored": 0, "signature": signature}

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
            logger.info("truth_subspace: %d anchors, no scoreable nodes", len(active_anchors))
            return {"anchors": len(active_anchors), "nodes_scored": 0, "signature": signature}

        # Step 6: EMBED node texts (batched) and compute coords per node.
        node_vecs = await _embed_in_batches(embedding_engine, node_texts)
        for node_id, node_vec in zip(node_ids, node_vecs):
            try:
                coords = align.node_coords(node_vec, anchor_vecs)
                scored[node_id] = coords
            except Exception as error:
                # Per-node fail-open: one bad node never sinks the batch.
                logger.debug("truth_subspace: coords failed for node %s: %s", node_id, error)

        if not scored:
            return {"anchors": len(active_anchors), "nodes_scored": 0, "signature": signature}

        # Step 7: PERSIST per-node coordinate vectors.
        try:
            await graph_engine.set_node_truth_alignments(scored)
        except Exception as error:
            logger.warning("truth_subspace: persisting alignments failed open: %s", error)
            return {"anchors": len(active_anchors), "nodes_scored": 0, "signature": signature}

    logger.info(
        "truth_subspace: built subspace -> anchors=%d nodes_scored=%d signature=%s",
        len(active_anchors),
        len(scored),
        signature,
    )
    return {
        "anchors": len(active_anchors),
        "nodes_scored": len(scored),
        "signature": signature,
    }
