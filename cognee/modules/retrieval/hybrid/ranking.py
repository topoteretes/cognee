from typing import Optional

from cognee.modules.retrieval.hybrid.results import payload, result_id
from cognee.modules.truth_subspace.align import truth_factor


def rank_chunk_summary_pairs(
    pairs: list[dict],
    limit: int,
    use_importance_weight: bool,
    use_truth_weight: bool = False,
    q_coords: Optional[list[float]] = None,
    truth_state_by_id: Optional[dict] = None,
    current_truth_epoch: Optional[int] = None,
) -> list[dict]:
    if limit <= 0:
        return []

    rrf_k = _rrf_k(limit)
    ranked = []
    for pair in pairs:
        chunk = pair["chunk"]
        if chunk is None:
            continue

        ranks = [
            rank
            for rank in (pair["bm25_rank"], pair["vector_rank"], pair["summary_rank"])
            if rank is not None
        ]
        if not ranks:
            continue

        rrf_score = sum(1.0 / (rrf_k + rank + 1) for rank in ranks)
        final_score = rrf_score
        if use_importance_weight:
            final_score *= _importance_factor(chunk)

        chunk_id = pair["chunk_id"] or result_id(chunk) or ""
        if use_truth_weight and q_coords and current_truth_epoch is not None:
            truth_state = (truth_state_by_id or {}).get(chunk_id, {})
            if truth_state.get("truth_epoch") == current_truth_epoch:
                final_score *= truth_factor(truth_state.get("truth_alignment", []), q_coords)

        ranked.append((final_score, rrf_score, min(ranks), chunk_id, pair))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [pair for *_, pair in ranked[:limit]]


def _rrf_k(chunks_top_k: int) -> int:
    return max(30, min(60, 20 + 2 * chunks_top_k))


def _importance_factor(chunk) -> float:
    raw_importance = payload(chunk).get("importance_weight")
    importance = raw_importance if isinstance(raw_importance, (int, float)) else 0.5
    importance = max(0.0, min(1.0, importance))
    return 0.75 + 0.5 * importance
