from cognee.modules.retrieval.hybrid.ranking import rank_chunk_summary_pairs


def _pair(chunk_id: str, rank: int) -> dict:
    return {
        "chunk": {"id": chunk_id, "text": chunk_id, "importance_weight": 0.5},
        "chunk_id": chunk_id,
        "vector_rank": rank,
        "summary_rank": None,
    }


def test_min_score_none_keeps_the_same_hits():
    pairs = [_pair("high", 0), _pair("low", 5)]

    baseline = rank_chunk_summary_pairs(pairs, limit=2, use_importance_weight=False)
    with_none = rank_chunk_summary_pairs(
        pairs, limit=2, use_importance_weight=False, min_score=None
    )

    assert [pair["chunk_id"] for pair in with_none] == [pair["chunk_id"] for pair in baseline]
    assert len(with_none) == 2


def test_min_score_keeps_hits_at_or_above_the_fused_score():
    # limit=2 => rrf_k=30. Rank 0 scores 1/31; rank 5 scores 1/36.
    ranked = rank_chunk_summary_pairs(
        [_pair("high", 0), _pair("low", 5)],
        limit=2,
        use_importance_weight=False,
        min_score=1 / 32,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["high"]


def test_min_score_keeps_a_hit_whose_fused_score_equals_the_cutoff():
    ranked = rank_chunk_summary_pairs(
        [_pair("exact", 0)],
        limit=2,
        use_importance_weight=False,
        min_score=1 / 31,
    )

    assert [pair["chunk_id"] for pair in ranked] == ["exact"]


def test_min_score_uses_importance_adjusted_score_not_raw_rrf():
    # limit=2 => rrf_k=30. Rank 0 raw RRF is 1/31. importance_weight=0
    # multiplies that by 0.75, so a raw-RRF threshold drops the hit.
    ranked = rank_chunk_summary_pairs(
        [
            {
                "chunk": {"id": "low-importance", "text": "x", "importance_weight": 0.0},
                "chunk_id": "low-importance",
                "vector_rank": 0,
                "summary_rank": None,
            }
        ],
        limit=2,
        use_importance_weight=True,
        min_score=1 / 31,
    )

    assert ranked == []


def test_min_score_returns_nothing_when_every_hit_is_below_the_cutoff():
    ranked = rank_chunk_summary_pairs(
        [_pair("a", 0), _pair("b", 1)],
        limit=2,
        use_importance_weight=False,
        min_score=1.0,
    )

    assert ranked == []
