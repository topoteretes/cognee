import math
from typing import Mapping, Optional

from cognee.modules.retrieval.hybrid.results import payload, result_id
from cognee.modules.truth_subspace.align import truth_factor


DEFAULT_CHANNEL_WEIGHTS = {
    "bm25": 1.0,
    "vector": 1.0,
    # A summary is derived from its source chunk, so it is supporting evidence
    # rather than an independent vote of equal strength.
    "summary": 0.5,
}

_CHANNEL_RANK_FIELDS = {
    "bm25": "bm25_rank",
    "vector": "vector_rank",
    "summary": "summary_rank",
}


def rank_chunk_summary_pairs(
    pairs: list[dict],
    limit: int,
    use_importance_weight: bool,
    use_truth_weight: bool = False,
    q_coords: Optional[list[float]] = None,
    truth_state_by_id: Optional[dict] = None,
    current_truth_epoch: Optional[int] = None,
    channel_weights: Optional[Mapping[str, float]] = None,
    rrf_k: Optional[int] = None,
) -> list[dict]:
    if limit <= 0:
        return []

    resolved_rrf_k = rrf_k if isinstance(rrf_k, int) and rrf_k > 0 else _rrf_k(limit)
    weights = _channel_weights(channel_weights)
    ranked = []
    for pair in pairs:
        chunk = pair["chunk"]
        if chunk is None:
            continue

        attribution = _channel_attribution(pair, weights, resolved_rrf_k)
        if not attribution:
            continue

        rrf_score = sum(channel["contribution"] for channel in attribution)
        final_score = rrf_score
        importance = _importance(chunk) if use_importance_weight else None

        chunk_id = pair["chunk_id"] or result_id(chunk) or ""
        applied_truth_factor = 1.0
        if use_truth_weight and q_coords and current_truth_epoch is not None:
            truth_state = (truth_state_by_id or {}).get(chunk_id, {})
            if truth_state.get("truth_epoch") == current_truth_epoch:
                applied_truth_factor = truth_factor(
                    truth_state.get("truth_alignment", []), q_coords
                )
                final_score *= applied_truth_factor

        ranked_pair = dict(pair)
        ranked_pair.update(
            {
                "retrieval_score": final_score,
                "rrf_score": rrf_score,
                "retrieval_channels": attribution,
                "importance_weight": importance,
                "truth_factor": applied_truth_factor,
            }
        )
        ranks = [channel["rank"] for channel in attribution]
        # Importance is intentionally only a tie-break. A query-independent
        # value must never promote a less relevant candidate over a stronger
        # lexical or semantic result.
        importance_tiebreak = importance if importance is not None else 0.5
        ranked.append(
            (
                final_score,
                rrf_score,
                importance_tiebreak,
                min(ranks),
                chunk_id,
                ranked_pair,
            )
        )

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    return [pair for *_, pair in ranked[:limit]]


def _rrf_k(chunks_top_k: int) -> int:
    return max(30, min(60, 20 + 2 * chunks_top_k))


def _importance(chunk) -> float:
    raw_importance = payload(chunk).get("importance_weight")
    importance = raw_importance if isinstance(raw_importance, (int, float)) else 0.5
    numeric_importance = float(importance)
    if not math.isfinite(numeric_importance):
        return 0.5
    return max(0.0, min(1.0, numeric_importance))


def _channel_weights(overrides: Optional[Mapping[str, float]]) -> dict[str, float]:
    weights = dict(DEFAULT_CHANNEL_WEIGHTS)
    for channel, weight in (overrides or {}).items():
        if channel not in weights or not isinstance(weight, (int, float)):
            continue
        numeric_weight = float(weight)
        if math.isfinite(numeric_weight) and numeric_weight >= 0:
            weights[channel] = numeric_weight
    return weights


def _channel_attribution(pair: dict, weights: Mapping[str, float], rrf_k: int) -> list[dict]:
    attribution = []
    for channel, rank_field in _CHANNEL_RANK_FIELDS.items():
        rank = pair.get(rank_field)
        weight = weights[channel]
        if not isinstance(rank, int) or rank < 0 or weight <= 0:
            continue

        item = {
            "channel": channel,
            "rank": rank,
            "weight": weight,
            "contribution": weight / (rrf_k + rank + 1),
        }
        native_score = pair.get(f"{channel}_score")
        if isinstance(native_score, (int, float)) and math.isfinite(float(native_score)):
            # Native scores are retained for observability. They are not mixed
            # across heterogeneous stores without explicit calibration.
            item["native_score"] = float(native_score)
        attribution.append(item)
    return attribution
