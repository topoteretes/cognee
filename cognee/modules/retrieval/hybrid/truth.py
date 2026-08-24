from typing import NamedTuple, Optional

from cognee.context_global_variables import current_dataset_id
from cognee.modules.retrieval.hybrid.chunks import chunk_candidate_limit, search_collection
from cognee.modules.retrieval.hybrid.results import result_id
from cognee.modules.truth_subspace import align
from cognee.modules.truth_subspace.centroids import load_centroids, pad_coords
from cognee.modules.truth_subspace.constants import DEFAULT_K
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


class TruthContext(NamedTuple):
    q_coords: Optional[list[float]] = None
    truth_state_by_id: Optional[dict] = None
    current_truth_epoch: Optional[int] = None


async def build_truth_context(
    unified_engine,
    query_vector: list[float],
    *,
    use_truth_weight: bool,
    chunks_top_k: int,
    node_name,
    node_name_filter_operator: str,
) -> TruthContext:
    """Truth-subspace alignment context for the chunk lane.

    Values are ``None`` when the truth weight is off or centroid slots are absent,
    so ranking stays at exact baseline. Fails open to baseline on any error.
    """
    if not use_truth_weight:
        return TruthContext()

    try:
        dataset_id = current_dataset_id.get()
        if dataset_id is None:
            return TruthContext()

        centroids = await load_centroids(unified_engine.vector, str(dataset_id))
        if not centroids:
            return TruthContext()

        centroid_vectors = [centroid.centroid for centroid in centroids]
        q_coords = pad_coords(align.query_coords(query_vector, centroid_vectors), DEFAULT_K)
        current_truth_epoch = max(centroid.truth_epoch for centroid in centroids)

        candidate_chunk_ids = await _candidate_chunk_ids(
            unified_engine.vector,
            query_vector,
            chunks_top_k=chunks_top_k,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
        )
        if not candidate_chunk_ids:
            return TruthContext(q_coords, {}, current_truth_epoch)

        truth_state_by_id = await unified_engine.graph.get_node_truth_state(candidate_chunk_ids)
        return TruthContext(q_coords, truth_state_by_id, current_truth_epoch)
    except Exception as error:
        logger.debug("Truth-subspace lookup failed; using baseline ranking: %s", error)
        return TruthContext()


async def _candidate_chunk_ids(
    vector_engine,
    query_vector: list[float],
    *,
    chunks_top_k: int,
    node_name,
    node_name_filter_operator: str,
) -> list[str]:
    """Candidate DocumentChunk ids whose truth alignments we batch-fetch.

    Mirrors the chunk lane's vector candidate window so the truth coords map
    covers the chunks that ranking can surface.
    """
    chunk_hits = await search_collection(
        vector_engine,
        "DocumentChunk_text",
        "",
        chunk_candidate_limit(chunks_top_k),
        node_name,
        node_name_filter_operator,
        query_vector=query_vector,
    )
    ids = []
    for hit in chunk_hits:
        chunk_id = result_id(hit)
        if chunk_id:
            ids.append(str(chunk_id))
    return ids
