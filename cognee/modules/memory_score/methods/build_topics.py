"""Cluster the tenant's graph into topics and gate on the data floor.

Two jobs, in this order:

1. **DATA FLOOR GATE** — evaluated before any generation or judging. The graph
   needs at least ``min_chunks`` document chunks, and clustering has to yield at
   least ``min_topics`` topics of at least ``min_nodes_per_topic`` nodes each. A
   failure returns immediately with ``below_data_floor=True`` plus a
   ``floor_reason``; nothing is generated and nothing is judged.

   The chunk half is free: one filtered graph read, no embeddings. The topic
   half is NOT free, and cannot be — deciding whether the graph clusters into
   enough topics requires the node vectors, and on every vector adapter except
   LanceDB ``fetch_node_embeddings`` cannot read stored vectors back and
   re-embeds instead (up to ``SEMANTIC_NODE_CAP`` nodes, one batched call). So a
   run that ends up gated on a topic shortfall has already paid for that one
   embedding batch. No LLM completion token is ever spent below the floor.
2. **Topics** — k-means over the node embeddings via
   :func:`cognee.modules.visualization.semantic_clusters.compute_clusters`.
   The clustering itself is pure numpy with a fixed seed, so topics are
   deterministic and cost nothing beyond the vectors above. Labels come from
   ``compute_clusters``' own labelling seam (``default_label``: top entities by
   degree/importance).

Real traffic then *weights* those topics: each real question is assigned to its
nearest topic by cosine similarity between the question embedding and the topic
centroid, which is what ``Topic.real_question_count`` records. No real question
is ever scored for correctness here — they have no golden answer.

``floor_reason`` is a raw signal, not a call to action: it states what the data
looks like ("38 document chunks, 50 required"), never what to do about it.
Deciding between "upload more data" and "define a schema" is the UI's job.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.modules.visualization.embedding_join import fetch_node_embeddings, select_nodes
from cognee.modules.visualization.preprocessor import preprocess
from cognee.modules.visualization.semantic_clusters import compute_clusters
from cognee.shared.logging_utils import get_logger

logger = get_logger("memory_score.build_topics")

# Graph node type that carries the source text a synthetic question is grounded in.
CHUNK_NODE_TYPE = "DocumentChunk"

_EPS = 1e-12


@dataclass
class Topic:
    """One cluster of the tenant's graph, plus how much real traffic hits it.

    ``node_ids`` are graph node ids (strings, as ``compute_clusters`` returns
    them) for the cluster's members. The ids of ``DocumentChunk`` nodes are
    listed FIRST, so a caller that needs the chunk text to ground a synthetic
    question can take the head of the list instead of type-probing every node.
    A cluster made purely of entities may contain no chunk id at all — callers
    must tolerate that.

    ``real_question_count`` is 0 when the tenant has no search history, and 0
    on every topic when the run is below the data floor (assignment needs
    embeddings, and a gated run spends nothing).
    """

    label: str
    node_ids: List[str]
    real_question_count: int


@dataclass
class TopicPlan:
    """The clustering result plus the data-floor verdict.

    When ``below_data_floor`` is True the caller must persist the run as
    SKIPPED_INSUFFICIENT_DATA and return early — no generation, no judging.
    ``topics`` is empty on a chunk shortfall (clustering never ran) and holds
    the surviving-but-too-few topics on a topic shortfall, since that is a more
    informative raw signal than an empty list.
    """

    topics: List[Topic]
    chunk_count: int
    below_data_floor: bool
    floor_reason: Optional[str]


async def _count_chunk_ids() -> List[str]:
    """Return the graph's ``DocumentChunk`` node ids.

    Counted on the graph, not on the relational ``Data`` table. ``Data`` has an
    indexed ``tenant_id`` and would be cheaper, but a ``Data`` row is a
    DOCUMENT, not a chunk — one document yields many chunks — so the two counts
    are not equivalent and a relational count cannot satisfy a chunk floor.
    There is no chunk-count column anywhere, and ``get_graph_metrics`` reports
    only whole-graph totals with no per-type breakdown.

    The ids are kept (the properties, including each chunk's full ``text``, are
    dropped) so the chunk-first ordering in :class:`Topic` costs no second fetch.
    """
    graph_engine = await get_graph_engine()
    nodes, _ = await graph_engine.get_filtered_graph_data([{"type": [CHUNK_NODE_TYPE]}])

    chunk_ids: List[str] = []
    for node in nodes:
        try:
            node_id, properties = node
        except (TypeError, ValueError):
            logger.warning("build_topics: skipping graph node with unexpected shape: %r", node)
            continue
        if isinstance(properties, dict) and properties.get("type") == CHUNK_NODE_TYPE:
            chunk_ids.append(str(node_id))
    return chunk_ids


async def _embedded_nodes() -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Renderer-shaped graph nodes plus their stored embeddings.

    Reuses the semantic-map pipeline verbatim: ``preprocess`` enriches raw
    adapter output with the ``degree`` / ``importance`` / ``name`` /
    ``is_unnamed`` fields ``compute_clusters``' labelling reads, ``select_nodes``
    bounds the set deterministically, and ``fetch_node_embeddings`` resolves
    vectors from the vector store (re-embedding only where the adapter cannot
    return them).
    """
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

    pre = preprocess(graph_data)
    nodes = select_nodes(pre.nodes)
    embeddings = await fetch_node_embeddings(nodes)
    return nodes, embeddings


def _chunks_first(node_ids: Sequence[str], chunk_ids: set) -> List[str]:
    """Order a cluster's members so ``DocumentChunk`` ids come first."""
    chunks = [nid for nid in node_ids if nid in chunk_ids]
    others = [nid for nid in node_ids if nid not in chunk_ids]
    return chunks + others


def _centroid(node_ids: Sequence[str], embeddings: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """Mean member vector, or None when no member of the cluster has one."""
    vectors = [embeddings[nid] for nid in node_ids if nid in embeddings]
    if not vectors:
        return None
    matrix = np.array(vectors, dtype=float)
    return matrix.mean(axis=0)


async def _weight_topics_by_real_traffic(
    topics: List[Topic],
    clusters: List[Dict[str, Any]],
    embeddings: Dict[str, List[float]],
    real_questions: List[str],
) -> None:
    """Assign every real question to its nearest topic, in place.

    Nearest = highest cosine similarity between the question embedding and the
    topic centroid. Ties go to the lowest topic index, so the assignment is
    deterministic. Best-effort: an embedding failure leaves every count at 0
    (topics then get an equal share of the synthetic budget) rather than
    failing the whole run.
    """
    centroids: List[Tuple[int, np.ndarray]] = []
    for index, cluster in enumerate(clusters):
        centroid = _centroid(cluster["node_ids"], embeddings)
        if centroid is not None:
            centroids.append((index, centroid))
    if not centroids:
        return

    try:
        embedding_engine = get_embedding_engine()
        question_vectors = await embedding_engine.embed_text(real_questions)
    except Exception as exc:  # a weighting failure must not fail the score run
        logger.warning(
            "build_topics: could not embed %d real question(s) (%s); "
            "topics keep real_question_count=0",
            len(real_questions),
            exc,
        )
        return

    matrix = np.array([centroid for _, centroid in centroids], dtype=float)
    unit_matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + _EPS)

    for question_vector in question_vectors:
        vector = np.array(question_vector, dtype=float)
        if vector.shape[0] != unit_matrix.shape[1]:
            # Node vectors and question vectors come from the same configured
            # engine; a mismatch means the embedding model changed since the
            # graph was built, and similarity would be meaningless.
            logger.warning(
                "build_topics: question embedding dim %d != node embedding dim %d; "
                "skipping real-traffic weighting",
                vector.shape[0],
                unit_matrix.shape[1],
            )
            return
        unit_vector = vector / (np.linalg.norm(vector) + _EPS)
        best = int(np.argmax(unit_matrix @ unit_vector))
        topics[centroids[best][0]].real_question_count += 1


async def build_topics(
    real_questions: List[str],
    min_chunks: int = 50,
    min_topics: int = 3,
    min_nodes_per_topic: int = 5,
) -> TopicPlan:
    """Gate on the data floor, then cluster the graph into weighted topics.

    Args:
        real_questions: the tenant's real past question texts, used only to
            weight topics. Empty is fine — every ``real_question_count`` is
            then 0, which is NOT a floor failure.
        min_chunks: minimum ``DocumentChunk`` nodes required in the graph.
        min_topics: minimum topics that must survive the size filter.
        min_nodes_per_topic: minimum nodes for a cluster to count as a topic.

    Returns:
        A :class:`TopicPlan`. ``below_data_floor=True`` means the caller stops
        here.

    Scope note: the graph engine is resolved from the ambient database context,
    so this clusters whatever graph the caller has entered (tenant-wide when the
    caller set the tenant's context).

    Cost note: no LLM completion call is made anywhere in here. The only paid
    calls are embeddings — resolving the node vectors for clustering (which
    re-embeds on adapters that cannot return stored vectors) and, after the gate
    has passed, embedding the real questions for weighting. The chunk half of the
    gate is evaluated before even that.
    """
    chunk_ids = await _count_chunk_ids()
    chunk_count = len(chunk_ids)

    if chunk_count < min_chunks:
        # Gate hit on the cheapest possible signal: no full-graph fetch, no
        # embeddings, no clustering, no LLM.
        return TopicPlan(
            topics=[],
            chunk_count=chunk_count,
            below_data_floor=True,
            floor_reason=(f"Graph has {chunk_count} document chunk(s); {min_chunks} required."),
        )

    nodes, embeddings = await _embedded_nodes()
    if not embeddings:
        return TopicPlan(
            topics=[],
            chunk_count=chunk_count,
            below_data_floor=True,
            floor_reason=(
                f"No node embeddings could be resolved for {len(nodes)} graph node(s), "
                "so the graph cannot be clustered into topics."
            ),
        )

    clusters = compute_clusters(nodes, embeddings)["clusters"]
    # compute_clusters already drops empty clusters; drop the too-small ones.
    kept = [cluster for cluster in clusters if cluster["size"] >= min_nodes_per_topic]

    chunk_id_set = set(chunk_ids)
    topics = [
        Topic(
            label=cluster["label"],
            node_ids=_chunks_first(cluster["node_ids"], chunk_id_set),
            real_question_count=0,
        )
        for cluster in kept
    ]

    if len(kept) < min_topics:
        return TopicPlan(
            topics=topics,
            chunk_count=chunk_count,
            below_data_floor=True,
            floor_reason=(
                f"Clustering yielded {len(kept)} topic(s) with at least "
                f"{min_nodes_per_topic} nodes (out of {len(clusters)} cluster(s)); "
                f"{min_topics} required."
            ),
        )

    # Blank history rows would embed to noise and be assigned to an arbitrary topic.
    questions = [question.strip() for question in real_questions if question and question.strip()]
    if questions:
        await _weight_topics_by_real_traffic(topics, kept, embeddings, questions)

    logger.info(
        "build_topics: %d chunk(s), %d topic(s), %d real question(s) weighted across topics",
        chunk_count,
        len(topics),
        sum(topic.real_question_count for topic in topics),
    )

    return TopicPlan(
        topics=topics,
        chunk_count=chunk_count,
        below_data_floor=False,
        floor_reason=None,
    )
